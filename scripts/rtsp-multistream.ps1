# Spawn multiple RTSP streams from a local MP4 for quick dashboard testing (Windows PowerShell).
# Usage: .\rtsp-multistream.ps1 [-Input data\samples\demo.mp4] [-Streams 4] [-Host 0.0.0.0] [-Port 8554]
# Requirements: ffmpeg in PATH; allow app through Windows firewall if needed.

param(
    [string]$Input = "data\samples\demo.mp4",
    [int]$Streams = 4,
    [string]$Host = "0.0.0.0",
    [int]$Port = 8554
)

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Error "ffmpeg not found in PATH. Please install ffmpeg first."
    exit 1
}

if (-not (Test-Path $Input)) {
    Write-Error "Input file not found: $Input"
    exit 1
}

Write-Host "Starting $Streams RTSP streams from $Input ..."

$procs = @()
for ($i = 1; $i -le $Streams; $i++) {
    $name = ("stream{0:D2}" -f $i)
    $args = @(
        "-hide_banner", "-loglevel", "warning",
        "-re", "-stream_loop", "-1",
        "-i", $Input,
        "-c", "copy",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        "-muxdelay", "0.1",
        "-listen", "1",
        "rtsp://$Host`:$Port/$name"
    )
    $proc = Start-Process -FilePath "ffmpeg" -ArgumentList $args -PassThru -WindowStyle Hidden
    $procs += $proc
}

Write-Host "RTSP endpoints:"
for ($i = 1; $i -le $Streams; $i++) {
    $name = ("stream{0:D2}" -f $i)
    Write-Host ("  rtsp://{0}:{1}/{2}" -f $Host, $Port, $name)
}

Write-Host "Press Enter to stop all streams..."
[void][Console]::ReadLine()

foreach ($p in $procs) {
    if (!$p.HasExited) { $p.Kill() }
}
Write-Host "Stopped."
