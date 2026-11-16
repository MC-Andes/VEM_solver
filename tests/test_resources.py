from vem3d import resources


def test_auto_processes_respect_slurm(monkeypatch):
    monkeypatch.setattr(resources, "nucleos_fisicos", lambda: 64)
    monkeypatch.setattr(resources, "memoria_disponible_gb", lambda: 256)
    monkeypatch.setattr(
        resources.os, "sched_getaffinity", lambda pid: set(range(64)), raising=False
    )
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "8192")
    assert resources.procesos_recomendados(1)[0] == 2
    assert resources.procesos_recomendados(4)[0] == 1
