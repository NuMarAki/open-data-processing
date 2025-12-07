"""Menu de processamento de bases"""
from .menu_base import MenuBase

class MenuProcessar(MenuBase):
    """Menu para processamento de bases de dados"""
    
    def exibir(self) -> str:
        """Exibe o menu de processamento"""
        self.exibir_secao("Processamento de Bases")
        self.exibir_opcoes()
        return self.obter_entrada("Escolha uma opção: ")
    
    def exibir_opcoes(self):
        """Exibe as opções de processamento"""
        opcoes = [
            ("1", "Processar PNAD", "Pesquisa Nacional por Amostra de Domicílios"),
            ("2", "Processar RAIS", "Relação Anual de Informações Sociais"),
            ("3", "Processar CAGED", "Cadastro Geral de Empregados e Desempregados"),
            ("4", "Processar TODAS", "Executar processamento completo (PNAD, RAIS, CAGED)"),
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
            '0': 'voltar'
        }
        return mapa.get(opcao, 'invalido')
