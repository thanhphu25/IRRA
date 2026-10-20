import pytest
import yaml
from utils.options import get_args


def test_config_restore_resume_and_cli_override(tmp_path):
    saved = vars(get_args(['--lgsp_mode', 'both', '--local_temperature', '0.3']))
    saved['init_checkpoint'] = 'old-best.pth'
    config = tmp_path / 'configs.yaml'
    config.write_text(yaml.safe_dump(saved), encoding='utf-8')
    args = get_args(['--resume', '--resume_ckpt_file', str(tmp_path / 'last.pth'), '--batch_size', '8'])
    assert args.lgsp_mode == 'both' and args.local_temperature == 0.3 and args.batch_size == 8
    assert not args.init_checkpoint


def test_no_ambiguous_warmstart_resume():
    with pytest.raises(SystemExit):
        get_args(['--resume', '--init_checkpoint', 'best.pth'])


def test_explicit_log_eval_period_types():
    args = get_args(['--log_period', '10', '--eval_period', '2'])
    assert args.log_period == 10 and args.eval_period == 2
