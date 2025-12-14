"""Classe base para menus interativos"""

class MenuBase:
    """Classe base para implementação de menus"""
    
    @staticmethod
    def exibir_titulo(titulo: str):
        """Exibe um título formatado"""
        print("\n" + "="*70)
        print(f"{titulo:^70}")
        print("="*70)
    
    @staticmethod
    def exibir_secao(secao: str):
        """Exibe um cabeçalho de seção"""
        print(f"\n--- {secao} ---")
    
    @staticmethod
    def linha_separadora():
        """Exibe uma linha separadora"""
        print("-" * 70)
    
    @staticmethod
    def obter_entrada(mensagem: str) -> str:
        """Obtém entrada do usuário"""
        return input(f"\n{mensagem}").strip()
    
    @staticmethod
    def obter_confirmacao(mensagem: str) -> bool:
        """Obtém confirmação sim/não do usuário"""
        resposta = input(f"\n{mensagem} (S/N): ").strip().upper()
        return resposta == 'S'
    
    @staticmethod
    def pausar():
        """Pausa aguardando enter"""
        input("\n[Pressione ENTER para continuar...]")
