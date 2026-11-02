# 单被试攻关流水线：生成逐人候选(ps/ps2/ps3) -> 诊断(等权二候选) -> 加权三候选+阈值偏置搜索；结果写 results.json
param([string]$q)
$py = "d:\WORK\BCIC2026\CogState\.venv\Scripts\python.exe"
$d = "d:\WORK\BCIC2026\CogState\test\16_subject_push"
$vs = @('', '--v2', '--v3'); foreach($x in 'v4','v5','v6'){ if ($args -contains $x) { $vs += "--$x" } }
foreach ($v in $vs) {
    $tag = @{ '' = 'scores_ps'; '--v2' = 'scores_ps2'; '--v3' = 'scores_ps3'; '--v4' = 'scores_ps4'; '--v5' = 'scores_ps5'; '--v6' = 'scores_ps6' }[$v]
    if (-not (Test-Path "$d\$tag\sub-$q.npz")) { if ($v) { & $py "$d\ps_cands.py" $q $v | Out-Null } else { & $py "$d\ps_cands.py" $q | Out-Null } }
}
$o = & $py "$d\diag.py" $q
$o[1]; $o[-1]
& $py "$d\wsearch.py" $q 16 --longh --bias | Select-Object -Last 2


