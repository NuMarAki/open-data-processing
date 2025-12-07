"""Menu principal do sistema"""
from .menu_base import MenuBase

class MenuPrincipal(MenuBase):
    """Menu principal de navegação"""
    
    TITULO = "ANÁLISE DE ETARISMO EM TI NO BRASIL"
    SUBTITULO = "TCC - MBA Data Science USP/Esalq"
    VERSAO = "0.5.30"
    
    def exibir(self) -> str:
        """Exibe o menu principal e retorna a opção escolhida"""
        self.exibir_titulo(f"{self.TITULO} v{self.VERSAO}")
        print(f"{self.SUBTITULO:^70}\n")
        
        self.exibir_opcoes()
        
        return self.obter_entrada("Escolha uma opção: ")
    
    def exibir_opcoes(self):
        """Exibe as opções do menu"""
        opcoes = [
            ("1", "Descompactação de Arquivos", "Extrair e organizar dados compactados"),
            ("2", "Processar Bases", "Executar processamento de dados"),
            ("3", "Relatórios e Análises", "Gerar gráficos e análises"),
            ("4", "Diagnóstico de Dados", "Analisar qualidade dos dados"),
            ("5", "Modelo Preditivo RAIS", "Treinar e avaliar modelo de previsão"),
            ("0", "Sair", "Encerrar aplicação"),
        ]
        
        for codigo, opcao, descricao in opcoes:
            print(f"  {codigo}. {opcao:<30} - {descricao}")
    
    def processar_opcao(self, opcao: str) -> str:
        """Retorna o tipo de submenu a exibir"""
        mapa_opcoes = {
            '1': 'descompactacao',
            '2': 'processar',
            '3': 'relatorios',
            '4': 'diagnostico',
            '5': 'preditivo',
            '0': 'sair'
        }
        return mapa_opcoes.get(opcao, 'invalido')
