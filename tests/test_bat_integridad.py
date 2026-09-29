"""
Integridad de los .bat: ningun caracter de CONTROL adentro.

POR QUE EXISTE (incidente 29/9/2026). `ft_run_diario.bat` quedo con cuatro bytes
0x0C (form feed) donde debia decir `scripts\\forward_testing\\ft_bot_smc_v3.py`:
el bloque se escribio desde Python con un string NO raw, y ahi `\\f` es un escape
valido que Python convierte en form feed ANTES de escribir el archivo. El .bat
tenia sus CRLF perfectos -- la regla conocida ("editar en binario, nunca con
sed") se habia cumplido -- y aun asi el path estaba roto.

Consecuencia: Python no encontraba el archivo, los dos bots de FT_SMC_v3 no
corrieron NUNCA desde su alta y la rutina informo "al menos un bot termino con
error" durante 6 corridas seguidas.

La regla que deja: para escribir un .bat desde Python no alcanza con abrir en
binario, el string TAMBIEN tiene que ser raw (r"..." / rb"..."). Los paths de
Windows estan llenos de escapes que Python interpreta sin avisar: \\f \\t \\n \\r
\\b \\a \\v \\0 y \\x. Este test es barato y los detecta a todos.
"""

import glob
import io
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Los que aparecerian si un string no-raw se comio un backslash. TAB queda afuera:
# es legitimo en un .bat (indentacion), y no lo produce ningun escape de path.
CONTROL = {
    0x00: r"NUL (\0)",
    0x07: r"BEL (\a)",
    0x08: r"BS (\b)",
    0x0B: r"VT (\v)",
    0x0C: r"FF (\f)",
    0x1B: "ESC",
}


def bats():
    return sorted(glob.glob(os.path.join(ROOT, "scripts", "**", "*.bat"), recursive=True))


def test_hay_bats_para_revisar():
    """Si el glob deja de encontrar archivos, el test de abajo pasaria vacio."""
    assert len(bats()) >= 20


@pytest.mark.parametrize("ruta", bats(), ids=lambda r: os.path.basename(r))
def test_sin_caracteres_de_control(ruta):
    crudo = io.open(ruta, "rb").read()
    hallados = {n: crudo.count(bytes([b])) for b, n in CONTROL.items()
                if crudo.count(bytes([b]))}
    assert not hallados, (
        f"{os.path.relpath(ruta, ROOT)} tiene caracteres de control {hallados}. "
        "Casi seguro se escribio desde Python con un string NO raw: un path como "
        r"'scripts\forward_testing' deja \f = 0x0C. Usar r'...' / rb'...'."
    )


def test_el_caso_exacto_del_incidente():
    """El path del bot de SMC_v3 en ft_run_diario.bat, byte a byte."""
    ruta = os.path.join(ROOT, "scripts", "manual", "ft_run_diario.bat")
    crudo = io.open(ruta, "rb").read()
    assert crudo.count(rb"scripts\forward_testing\ft_bot_smc_v3.py") == 2
    assert b"\x0c" not in crudo
