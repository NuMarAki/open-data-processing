"""Package de menus interativos para o sistema de análise"""
from .menu_base import MenuBase
from .menu_principal import MenuPrincipal
from .menu_processar import MenuProcessar
from .menu_relatorios import MenuRelatorios
from .menu_descompactacao import MenuDescompactacao

__all__ = [
    'MenuBase',
    'MenuPrincipal',
    'MenuProcessar',
    'MenuRelatorios',
    'MenuDescompactacao'
]
