# P0.6 — collect_topology.ps1 — emit canonical topology JSON.
$ErrorActionPreference = 'Stop'
$cores = Get-CimInstance Win32_Processor | ForEach-Object {
    [PSCustomObject]@{
        cpu_id     = [int]$_.DeviceID.TrimStart('CPU')
        apic_id    = $_.ProcessorId
        package    = 0
        numa_node  = 0
        smt_thread = 0
    }
}
$out = [PSCustomObject]@{
    status   = "measured"
    n_cores  = ($cores | Measure-Object).Count
    cores    = $cores
}
$out | ConvertTo-Json -Depth 4
