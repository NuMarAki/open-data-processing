"""Menu de descompactação de arquivos"""
from .menu_base import MenuBase

class MenuDescompactacao(MenuBase):
    """Menu para descompactação de arquivos"""
    
    def exibir(self) -> str:
        """Exibe o menu de descompactação"""
        self.exibir_secao("Descompactação de Arquivos")
        self.exibir_opcoes()
        return self.obter_entrada("Escolha uma opção: ")
    
    def exibir_opcoes(self):
        """Exibe as opções de descompactação"""
        opcoes = [
            ("1", "Descompactar PNAD", "Extrair arquivos da PNAD"),
            ("2", "Descompactar RAIS", "Extrair arquivos da RAIS"),
            ("3", "Descompactar CAGED", "Extrair arquivos do CAGED"),
            ("4", "Descompactar TODAS", "Extrair todas as bases"),
            ("5", "Listar Erros", "Exibir arquivos com erro de extração"),
            ("0", "Voltar", "Retornar ao menu anterior"),
        ]
        
        for codigo, opcao, descricao in opcoes:
            print(f"  {codigo}. {opcao:<20} - {descricao}")
    
    def processar_opcao(self, opcao: str) -> str:
        """Mapeia opção para base de dados"""
        mapa = {
            '1': 'pnad',
            '2': 'rais',
            '3': 'caged',
            '4': 'todas',
            '5': 'erros',
            '0': 'voltar'
        }
        return mapa.get(opcao, 'invalido')
