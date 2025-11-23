# Spawn multiple RTSP streams from a local MP4 for quick dashboard testing (Windows PowerShell).
# Usage: .\rtsp-multistream.ps1 [-Input data\samples\demo.mp4] [-Streams 4] [-Host 0.0.0.0] [-Port 8554]
# Requirements: ffmpeg in PATH; allow app through Windows firewall if needed.

param(
    [string]$Input = "",
    [int]$Streams = 4,
    [string]$ListenHost = "0.0.0.0",
    [int]$PortStart = 8554  # each stream uses PortStart + i - 1
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if ([string]::IsNullOrWhiteSpace($Input)) {
    $Input = Join-Path $scriptDir "..\data\samples\demo.mp4"
}
$Input = (Resolve-Path -Path $Input -ErrorAction Stop).Path

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Error "ffmpeg not found in PATH. Please install ffmpeg first."
    exit 1
}

Write-Host "Starting $Streams RTSP streams from $Input ..."

$procs = @()
for ($i = 1; $i -le $Streams; $i++) {
    $name = ("stream{0:D2}" -f $i)
    $port = $PortStart + $i - 1
    $args = @(
        "-hide_banner", "-loglevel", "warning",
        "-re", "-stream_loop", "-1",
        "-i", $Input,
        "-c", "copy",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        "-muxdelay", "0.1",
        "-listen", "1",
        "rtsp://$ListenHost`:$port/$name"
    )
    $proc = Start-Process -FilePath "ffmpeg" -ArgumentList $args -PassThru -WindowStyle Hidden
    $procs += $proc
}

Write-Host "RTSP endpoints:"
for ($i = 1; $i -le $Streams; $i++) {
    $name = ("stream{0:D2}" -f $i)
    $port = $PortStart + $i - 1
    Write-Host ("  rtsp://{0}:{1}/{2}" -f $ListenHost, $port, $name)
}

Write-Host "Press Enter to stop all streams..."
[void][Console]::ReadLine()

foreach ($p in $procs) {
    if (!$p.HasExited) { $p.Kill() }
}
Write-Host "Stopped."
