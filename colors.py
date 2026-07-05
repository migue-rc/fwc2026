from IPython.display import Markdown

def green_bold(s):
    return Markdown(f'<span style="color:green, font-weight:bold">{s}</span>')
def red_bold(s):
    return Markdown(f'<span style="color:red; font-weight:bold">{s}</span>')
def blue_bold(s):
    return Markdown(f'<span style="color:blue; font-weight:bold">{s}</span>')
def yellow_bold(s):
    return Markdown(f'<span style="color:yellow; font-weight:bold">{s}</span>')
def purple_bold(s):
    return Markdown(f'<span style="color:purple; font-weight:bold">{s}</span>')
def cyan_bold(s):
    return Markdown(f'<span style="color:cyan; font-weight:bold">{s}</span>')
def bold(s):
    return Markdown(f'<span style="font-weight:bold">{s}</span>')
def green(s):
    return Markdown(f'<span style="color:green">{s}</span>')
def red(s):
    return Markdown(f'<span style="color:red">{s}</span>')
def blue(s):
    return Markdown(f'<span style="color:blue">{s}</span>')
def yellow(s):
    return Markdown(f'<span style="color:yellow">{s}</span>')
def purple(s):
    return Markdown(f'<span style="color:purple">{s}</span>')
def cyan(s):
    return Markdown(f'<span style="color:cyan">{s}</span>')