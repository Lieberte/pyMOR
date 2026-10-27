$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$messagePath = Join-Path $repositoryRoot ".lark-completion-message.txt"
$cliPath = "C:\Users\shonc\AppData\Roaming\npm\node_modules\@larksuite\cli\scripts\run.js"
$recipientOpenId = "ou_595386bbc68fb631025c5127297a7d39"

if (-not (Test-Path -LiteralPath $messagePath)) {
    throw "Missing .lark-completion-message.txt"
}

$message = Get-Content -Raw -Encoding utf8 -LiteralPath $messagePath
$firstLine = ($message -split "\r?\n", 2)[0]
if ($firstLine -notmatch "pyMOR") {
    throw "The first line must identify a pyMOR completion notification"
}
if ($message.Length -gt 2000) {
    throw "Completion notification exceeds 2000 characters"
}

$sha256 = [System.Security.Cryptography.SHA256]::Create()
try {
    $hashBytes = $sha256.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($message))
} finally {
    $sha256.Dispose()
}
$hash = -join ($hashBytes[0..19] | ForEach-Object { $_.ToString("x2") })
$idempotencyKey = "pymor-$hash"

$env:LARKSUITE_CLI_NO_UPDATE_NOTIFIER = "1"
$env:LARKSUITE_CLI_NO_SKILLS_NOTIFIER = "1"
& node $cliPath --profile cursor im +messages-send `
    --user-id $recipientOpenId `
    --text $message `
    --as bot `
    --idempotency-key $idempotencyKey

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Remove-Item -LiteralPath $messagePath -Force
