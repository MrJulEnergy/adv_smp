import ipsuite as ips
import mlipx
import znmdakit

import apax.nodes as apax

from src import CompareDFT

project = ips.Project()

thermostat = ips.LangevinThermostat(time_step=..., temperature=..., friction=...)

with project:
    # 1. Build an initial box of liquid methanol
    methanol = ips.Smiles2Conformers(
        smiles="CO",
        numConfs=100,
    )
    box = ips.MultiPackmol(
        data=[methanol.frames],
        count=[32],
        density=792,  # kg/m^3, liquid methanol at ~298 K
        n_configurations=1,
    )

    # 2. Run a MD simulation with the MACE foundation model
    mace_calc = mlipx.GenericASECalculator(
        module="mace.calculators",
        class_name="mace_mp",
        device="auto",
        kwargs={
            "model": "/work/jpeters/models/mace-mpa-0-medium.model",
            "dispersion": True,
        },
    )

    mace_md = ips.ASEMD(
        data=box.frames,
        data_id=-1,
        model=mace_calc,
        thermostat=thermostat,
        steps=...,
        sampling_rate=...,
    )

    # 3. MGRID convergence test for the CP2K setup
    with project.group("mgrid"):
        sample = ips.RandomSelection(data=mace_md.frames, n_configurations=1)
        dft_a = ips.CP2KSinglePoint(
            data=sample.frames,
            cp2k_params="config/cp2k.yaml",
            cp2k_files=["config/BASIS_MOLOPT", "config/GTH_POTENTIALS"],
        )
        dft_b = ips.CP2KSinglePoint(...)
        compare = CompareDFT(
            data=[dft_a.frames, dft_b.frames],
            mgrids=[..., ...],
        )

    # 4. Generate the labelled dataset (DFT + D3 dispersion correction)
    selection = ips.RandomSelection(data=mace_md.frames, n_configurations=...)
    cp2k_model = ips.CP2KModel(
        config="config/cp2k.yaml",
        files=["config/BASIS_MOLOPT", "config/GTH_POTENTIALS"],
    )
    cp2k_data = ips.ApplyCalculator(
        data=selection.frames,
        model=cp2k_model,
        dump_rate=1,
    )
    d3_calc = ips.TorchDFTD3(
        xc=...,
        damping=...,
        cutoff=...,
        cnthr=...,
        dtype=...,
        skin=...,
    )
    dataset = ips.ApplyCalculator(
        data=cp2k_data.frames,
        model=d3_calc,
        dump_rate=1,
        additive=True,
    )

    # 5. Split the dataset into a fixed test set and a training set
    test_data = ips.RandomSelection(data=dataset.frames, n_configurations=...)
    train_data = ips.RandomSelection(
        data=test_data.excluded_frames,
        n_configurations=...,
    )

    # 6. Train an apax MLIP and evaluate it on the test set
    model = apax.Apax(
        data=train_data.frames,
        validation_data=test_data.frames,
        config="config/apax.yaml",
    )
    pred = ips.Prediction(data=test_data.frames, model=model)
    metrics = ips.PredictionMetrics(x=test_data.frames, y=pred.frames)

    # 7. Production MD with the trained MLIP
    large_box = ips.MultiPackmol(
        data=[methanol.frames],
        count=[64],
        density=792,
        n_configurations=1,
    )

    geom_opt = ips.ASEGeoOpt(
        data=large_box.frames,
        model=model,
        optimizer="BFGS",
        run_kwargs={"fmax": 0.05},
    )
    apax_md = apax.ApaxJaxMD(
        data=geom_opt.frames,
        data_id=-1,
        model=model,
        config="config/md.yaml",
    )

    # 8. Post-processing: RDF, MSD and self-diffusion
    u = znmdakit.Universe(
        data=apax_md.frames,
        residues={"MeO": "CO"},
    )
    rdf = znmdakit.InterRDF(
        universe=u.universe,
        g1="name C",
        g2="name O",
        nbins=...,
    )
    msd = znmdakit.EinsteinMSD(
        universe=u.universe,
        select="name COM and resname MeO",
        timestep=...,
        sampling_rate=...,
        apply_com_transform=True,
    )
    diff = znmdakit.SelfDiffusionFromMSD(
        data=msd,
        start_time=...,
        end_time=...,
    )

project.build()
