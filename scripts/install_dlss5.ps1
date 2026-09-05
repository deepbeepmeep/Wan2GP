[CmdletBinding()]
param(
    [string]$WanGPRoot,
    [switch]$AcceptThirdPartyRisk,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "DLSS 5 installation is supported on Windows only."
}

if (-not $WanGPRoot) {
    $WanGPRoot = Split-Path -Parent $PSScriptRoot
}
$WanGPRoot = (Resolve-Path -LiteralPath $WanGPRoot).Path
if (-not (Test-Path -LiteralPath (Join-Path $WanGPRoot "wgp.py") -PathType Leaf)) {
    throw "WanGP root not found at '$WanGPRoot' (wgp.py is missing)."
}

Write-Host "WanGP optional DLSS 5 installer" -ForegroundColor Cyan
Write-Warning @"
This installs native third-party binaries with GPU and filesystem access. The RenoDX
and DLSSNR downloads are community-hosted, not official NVIDIA/RenoDX releases.
DLSSNR 310.8.SF-v2 is NVIDIA-derived, modified, proprietary, and unsigned. WanGP
does not grant redistribution rights or guarantee these files. Continue only if you
accept the copyright, licensing, and security risks and their use is legal for you.
"@

if (-not $AcceptThirdPartyRisk) {
    $answer = Read-Host "Type I ACCEPT to download and install these components"
    if ($answer -cne "I ACCEPT") {
        throw "Installation cancelled."
    }
}

[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
Add-Type -AssemblyName System.IO.Compression

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
}

function Download-Verified([pscustomobject]$Package, [string]$Directory) {
    $destination = Join-Path $Directory $Package.File
    Write-Host "Downloading $($Package.Name)..."
    Invoke-WebRequest -Uri $Package.Url -OutFile $destination -UseBasicParsing -Headers @{ "User-Agent" = "WanGP-DLSS5-Installer" }
    $actual = Get-Sha256 $destination
    if ($actual -ne $Package.Sha256) {
        Remove-Item -LiteralPath $destination
        throw "$($Package.Name) checksum mismatch. Expected $($Package.Sha256), received $actual."
    }
    Write-Host "  verified $actual" -ForegroundColor DarkGreen
    return $destination
}

function Extract-ReShade64([string]$SetupPath, [string]$DestinationPath, [string]$TemporaryDirectory) {
    $payloadPath = Join-Path $TemporaryDirectory "reshade-payload.zip"
    $input = [IO.File]::OpenRead($SetupPath)
    $found = $false
    try {
        $block = New-Object byte[] 512
        while (($read = $input.Read($block, 0, $block.Length)) -ge 4) {
            if ($block[0] -ne 0x50 -or $block[1] -ne 0x4B -or $block[2] -ne 0x03 -or $block[3] -ne 0x04) {
                continue
            }
            $validHeader = $false
            for ($index = 4; $index -lt [Math]::Min(30, $read); $index++) {
                if ($block[$index] -ne 0) {
                    $validHeader = $true
                    break
                }
            }
            if (-not $validHeader) {
                continue
            }
            $output = [IO.File]::Create($payloadPath)
            try {
                $output.Write($block, 0, $read)
                $input.CopyTo($output)
            }
            finally {
                $output.Dispose()
            }
            $found = $true
            break
        }
    }
    finally {
        $input.Dispose()
    }
    if (-not $found) {
        throw "The verified ReShade setup does not contain an embedded payload."
    }

    $payload = [IO.File]::OpenRead($payloadPath)
    try {
        $archive = [IO.Compression.ZipArchive]::new($payload, [IO.Compression.ZipArchiveMode]::Read)
        try {
            $entry = $archive.GetEntry("ReShade64.dll")
            if ($null -eq $entry) {
                throw "ReShade64.dll is missing from the verified ReShade setup."
            }
            $source = $entry.Open()
            $output = [IO.File]::Create($DestinationPath)
            try {
                $source.CopyTo($output)
            }
            finally {
                $output.Dispose()
                $source.Dispose()
            }
        }
        finally {
            $archive.Dispose()
        }
    }
    finally {
        $payload.Dispose()
    }
}

function New-InstallItem([string]$Source, [string]$RelativeDestination, [string]$ExpectedHash = "") {
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Expected package file is missing: $Source"
    }
    if (-not $ExpectedHash) {
        $ExpectedHash = Get-Sha256 $Source
    }
    $actual = Get-Sha256 $Source
    if ($actual -ne $ExpectedHash) {
        throw "Extracted file checksum mismatch for '$Source'. Expected $ExpectedHash, received $actual."
    }
    return [pscustomobject]@{ Source = $Source; Destination = Join-Path $WanGPRoot (Join-Path "dlss5" $RelativeDestination); Sha256 = $ExpectedHash }
}

$packages = @(
    [pscustomobject]@{ Id = "workers"; Name = "WanGP DLSS 5 workers v1.1.3"; File = "WanGP-DLSS5-workers-v1.1.3.zip"; Url = "https://github.com/DeepBeepMeep/dlss5-visual-enhancer/releases/download/wangp-v1.1.3/WanGP-DLSS5-workers-v1.1.3.zip"; Sha256 = "EC470D8EB990CC04FE142C037B2F9E84C1D59A70B111DF51F110767897F5B0C2" }
    [pscustomobject]@{ Id = "reshade"; Name = "ReShade 6.8.0 full add-on setup"; File = "ReShade_Setup_6.8.0_Addon.exe"; Url = "https://reshade.me/downloads/ReShade_Setup_6.8.0_Addon.exe"; Sha256 = "AFE4C8F13048306307983B8B3D41D5BF00A86820440B0E57DEA10950E1176445" }
    [pscustomobject]@{ Id = "renodx"; Name = "RenoDX DLSS5 4.70"; File = "renodx-dlss5_4.70.zip"; Url = "https://github.com/RankFTW/rhi-repo/releases/download/renodx-dlss5-4.70/renodx-dlss5_4.70.zip"; Sha256 = "D6E356D01B429AF6288F488A4926C44F1D779A7D4586EE8C79D04D3A09A536E6" }
    [pscustomobject]@{ Id = "dlssnr"; Name = "DLSSNR 310.8.SF-v2"; File = "nvngx_dlssnr_310.8.SF-v2.zip"; Url = "https://github.com/RankFTW/rhi-repo/releases/download/dlssnr-310.8.SF-v2/nvngx_dlssnr_310.8.SF-v2.zip"; Sha256 = "1DA35941894994EB087E017577829E492454E9BAE3A6A9397027069CEB74955C" }
    [pscustomobject]@{ Id = "dlss"; Name = "DLSS Super Resolution 310.8.0"; File = "nvngx_dlss_310.8.0.zip"; Url = "https://github.com/RankFTW/rhi-repo/releases/download/dlss-310.8.0/nvngx_dlss_310.8.0.zip"; Sha256 = "FB481660F7E952B87F91760E3AFD7F9DC14CD2C3361B470E948D6346E4323009" }
    [pscustomobject]@{ Id = "dlssg"; Name = "DLSS Frame Generation 310.7.0"; File = "nvngx_dlssg_310.7.0.zip"; Url = "https://github.com/RankFTW/rhi-repo/releases/download/dlssg-310.7.0/nvngx_dlssg_310.7.0.zip"; Sha256 = "BFA977FB4451718C7D4A2217518DFC1AD30D77CE0EA026253C82BE96F5B9D35A" }
)

$temporaryBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$temporaryRoot = Join-Path $temporaryBase ("WanGP-DLSS5-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $temporaryRoot | Out-Null

try {
    $downloads = @{}
    foreach ($package in $packages) {
        $downloads[$package.Id] = Download-Verified $package $temporaryRoot
    }

    $staging = Join-Path $temporaryRoot "staging"
    $workers = Join-Path $staging "workers"
    $renodx = Join-Path $staging "renodx"
    $dlssnr = Join-Path $staging "dlssnr"
    $dlss = Join-Path $staging "dlss"
    $dlssg = Join-Path $staging "dlssg"
    foreach ($directory in @($workers, $renodx, $dlssnr, $dlss, $dlssg)) {
        New-Item -ItemType Directory -Path $directory | Out-Null
    }
    Expand-Archive -LiteralPath $downloads.workers -DestinationPath $workers
    Expand-Archive -LiteralPath $downloads.renodx -DestinationPath $renodx
    Expand-Archive -LiteralPath $downloads.dlssnr -DestinationPath $dlssnr
    Expand-Archive -LiteralPath $downloads.dlss -DestinationPath $dlss
    Expand-Archive -LiteralPath $downloads.dlssg -DestinationPath $dlssg

    $reshade64 = Join-Path $staging "ReShade64.dll"
    Extract-ReShade64 $downloads.reshade $reshade64 $temporaryRoot

    $installItems = @(
        New-InstallItem (Join-Path $workers "host\nr-depth-worker.exe") "host\nr-depth-worker.exe" "F8E2967912E5D596E8E36049370487B83620B0CB5845937B681CF835BAFC6D0B"
        New-InstallItem (Join-Path $workers "host\nvngx.dll") "host\nvngx.dll" "58191F4D38288C6BFBDA47EF56911D32052A9789E65714F4583F426E01464638"
        New-InstallItem (Join-Path $workers "dlssg\dlssg-worker.exe") "dlssg\dlssg-worker.exe" "D93084633E0AAB4A08C43A5EE240176716EF73D87F06F35C2293509FBFC8BD00"
        New-InstallItem $reshade64 "host\dxgi.dll" "0CEE63F9C9F13F3AC909C5B4903F4DBB4B719A7AB3B4F13B0DEAF83C814B94F7"
        New-InstallItem (Join-Path $renodx "renodx-dlss5.addon64") "host\renodx-dlss5.addon64" "D5ADF82EB44B065F4C590AC91FE824BAB07AFEA0EB9F994BDE936710C8593952"
        New-InstallItem (Join-Path $dlssnr "nvngx_dlssnr.dll") "host\nvngx_dlssnr.dll" "6EB209E764F39872625DEBD6ABAF45E2BB6322F6F270F781F70C059AE30B3927"
        New-InstallItem (Join-Path $dlss "nvngx_dlss.dll") "dlss\nvngx_dlss.dll" "C85F971CE023C9F3492FC7455F0B01A24BA18EA39636407A846902C4360B0B7E"
        New-InstallItem (Join-Path $dlssg "nvngx_dlssg.dll") "dlssg\nvngx_dlssg.dll" "135EAF0733C1E37381A8C28ABCF7A862404A54132B81787C04E35D09EFC5E36F"
        New-InstallItem (Join-Path $workers "LICENSE-DLSS5-Feeder.txt") "LICENSE-DLSS5-Feeder.txt"
        New-InstallItem (Join-Path $workers "LICENSE-Merserk.txt") "LICENSE-Merserk.txt"
        New-InstallItem (Join-Path $workers "LICENSE-WanGP-Adapters.txt") "LICENSE-WanGP-Adapters.txt"
        New-InstallItem (Join-Path $workers "README.txt") "README.txt"
        New-InstallItem (Join-Path $workers "SHA256SUMS.txt") "SHA256SUMS.txt"
    )

    $conflicts = @($installItems | Where-Object { (Test-Path -LiteralPath $_.Destination -PathType Leaf) -and (Get-Sha256 $_.Destination) -ne $_.Sha256 })
    if ($conflicts.Count -and -not $Force) {
        Write-Host "The following files already exist with different checksums:" -ForegroundColor Yellow
        $conflicts | ForEach-Object { Write-Host "  $($_.Destination)" }
        throw "No files were changed. Stop WanGP and rerun with -Force to back up and replace these files."
    }

    $backupStamp = Get-Date -Format "yyyyMMdd-HHmmssfff"
    foreach ($item in $installItems) {
        if (Test-Path -LiteralPath $item.Destination -PathType Leaf) {
            if ((Get-Sha256 $item.Destination) -eq $item.Sha256) {
                Write-Host "Already installed: $($item.Destination)"
                continue
            }
            $backup = "$($item.Destination).backup-$backupStamp"
            Copy-Item -LiteralPath $item.Destination -Destination $backup
            Write-Host "Backed up: $backup" -ForegroundColor Yellow
        }
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $item.Destination) | Out-Null
        Copy-Item -LiteralPath $item.Source -Destination $item.Destination -Force
        if ((Get-Sha256 $item.Destination) -ne $item.Sha256) {
            throw "Installed file verification failed: $($item.Destination)"
        }
        Write-Host "Installed: $($item.Destination)" -ForegroundColor Green
    }

    $signedNvidiaFiles = @(
        Join-Path $WanGPRoot "dlss5\dlss\nvngx_dlss.dll"
        Join-Path $WanGPRoot "dlss5\dlssg\nvngx_dlssg.dll"
    )
    foreach ($path in $signedNvidiaFiles) {
        $signature = Get-AuthenticodeSignature -LiteralPath $path
        if ($signature.Status -ne [Management.Automation.SignatureStatus]::Valid -or $signature.SignerCertificate.Subject -notmatch "NVIDIA") {
            throw "NVIDIA signature verification failed after installation: $path"
        }
    }

    Write-Host "DLSS 5 components are installed in '$WanGPRoot\dlss5'. Restart WanGP to refresh the available modes." -ForegroundColor Cyan
}
finally {
    $resolvedTemporaryRoot = [IO.Path]::GetFullPath($temporaryRoot)
    if ($resolvedTemporaryRoot.StartsWith($temporaryBase, [StringComparison]::OrdinalIgnoreCase) -and (Split-Path -Leaf $resolvedTemporaryRoot).StartsWith("WanGP-DLSS5-")) {
        Remove-Item -LiteralPath $resolvedTemporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
