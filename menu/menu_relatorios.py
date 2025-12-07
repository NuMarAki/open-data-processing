"""Menu de relatórios e análises"""
from .menu_base import MenuBase

class MenuRelatorios(MenuBase):
    """Menu para geração de relatórios e análises"""
    
    def exibir(self) -> str:
        """Exibe o menu de relatórios"""
        self.exibir_secao("Relatórios e Análises")
        self.exibir_opcoes()
        return self.obter_entrada("Escolha uma opção: ")
    
    def exibir_opcoes(self):
        """Exibe as opções de relatórios"""
        opcoes = [
            ("1", "Análise PNAD", "Relatórios e gráficos da PNAD"),
            ("2", "Análise RAIS", "Relatórios e gráficos da RAIS"),
            ("3", "Análise CAGED", "Relatórios e gráficos do CAGED"),
            ("4", "Análise Consolidada", "Relatórios e gráficos de todas as bases"),
            ("5", "Gráficos PNAD Específicos", "Escolaridade, renda, estado, série temporal"),
            ("0", "Voltar", "Retornar ao menu anterior"),
        ]
        
        for codigo, opcao, descricao in opcoes:
            print(f"  {codigo}. {opcao:<25} - {descricao}")
    
    def processar_opcao(self, opcao: str) -> str:
        """Mapeia opção para tipo de análise"""
        mapa = {
            '1': 'pnad',
            '2': 'rais',
            '3': 'caged',
            '4': 'consolidada',
            '5': 'graficos_pnad',
            '0': 'voltar'
        }
        return mapa.get(opcao, 'invalido')
