"""
可点击 Metric 卡片组件 (Streamlit declare_component)
用法: clicked = clickable_metric(key="A-600519", label="茅台", value="1520.50", delta="+2.3%")
"""
import streamlit.components.v1 as components


def clickable_metric(key: str, label: str, value: str,
                     delta: str = "", dark: bool = False,
                     height: int = 95) -> str | None:
    bg = "#161b22" if dark else "#fff"
    fg = "#c9d1d9" if dark else "#212529"
    border = "#30363d" if dark else "#dee2e6"
    if delta.startswith("+"):
        delta_color = "#26a69a"
    elif delta.startswith("-"):
        delta_color = "#ef5350"
    else:
        delta_color = "#8b949e" if dark else "#6c757d"

    html = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,sans-serif;">
<div id="card" style="background:{bg}; color:{fg}; border:1px solid {border};
border-radius:12px; padding:14px 10px; cursor:pointer;
text-align:center; box-sizing:border-box; width:100%; min-height:85px;
display:flex; flex-direction:column; justify-content:center;
" onclick="window.parent.postMessage({{isStreamlitMessage:true,type:'streamlit:setComponentValue',value:'{key}'}},'*'); this.style.boxShadow='0 2px 12px rgba(31,111,235,0.3)'; this.style.borderColor='#1f6feb'">
<div style="font-size:0.78rem;color:#8b949e;margin-bottom:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{label}</div>
<div style="font-size:1.25rem;font-weight:700;margin:2px 0">{value}</div>
<div style="font-size:0.82rem;color:{delta_color};font-weight:500">{delta}</div>
</div></body></html>"""
    result = components.html(html, height=height, scrolling=False)
    if isinstance(result, str) and result == key:
        return key
    return None
