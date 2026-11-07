"""Embed passive-end initialization in the single-file real entry."""
from pathlib import Path
from build_real_feeding_core import declaration

ROOT=Path(__file__).resolve().parents[1]
BEGIN,END='# BEGIN PASSIVE_END_RESET_CORE','# END PASSIVE_END_RESET_CORE'


def main():
    target=ROOT/'codes/Rolling_Arc_Real.py'
    text=target.read_text(encoding='utf-8')
    body=declaration(ROOT/'snake_control/passive_end_reset.py')
    if BEGIN in text:
        before,rest=text.split(BEGIN,1);_,after=rest.split(END,1)
    else:
        anchor='# ==================== 4. 飞特串口协议'
        before,after=text.split(anchor,1);after='\n\n'+anchor+after
    target.write_text(before+BEGIN+'\n'+body+'\n'+END+after,encoding='utf-8')


if __name__=='__main__':main()
