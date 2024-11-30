from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import random

app = Flask(__name__)

# Caminho do banco de dados
DB_PATH = "chamados.db"

# Inicializa o banco de dados
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chamados (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                categoria TEXT NOT NULL,
                prioridade TEXT NOT NULL,
                descricao TEXT NOT NULL,
                status TEXT DEFAULT 'Aberto',
                data_abertura TEXT DEFAULT CURRENT_TIMESTAMP,
                data_fechamento TEXT
            )
        """)
        conn.commit()
        
        # Exclui os dados antigos antes de inserir os novos
        excluir_dados_antigos()

        # Inserir dados de exemplo com datas aleatórias
        inserir_dados_exemplo()

def excluir_dados_antigos():
    """Exclui todos os dados antigos na tabela chamados"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chamados")
        conn.commit()

def inserir_dados_exemplo():
    """Insere dados de exemplo com datas de abertura aleatórias e mais variações"""
    nomes = [
        "Carlos Silva", "Maria Oliveira", "Pedro Santos", "Ana Costa", "Juliana Pereira",
        "Roberto Alves", "Lucas Martins", "Fernanda Rocha", "Marcos Souza", "Raquel Lima",
        "José Silva", "Patricia Pereira", "João Souza", "Vera Costa", "Gustavo Ferreira",
        "Tatiane Martins", "Fabio Oliveira", "Cláudia Rocha", "Leandro Santos", "Simone Lima"
    ]
    categorias = ["Rede", "Software", "Hardware"]
    prioridades = ["Alta", "Média", "Baixa"]
    descricoes = [
        "Problema com a conexão de internet", "Erro ao abrir o programa de vendas", "Computador lento",
        "Não consigo acessar a rede interna", "Atualização do sistema não finaliza", "Teclado não responde",
        "A rede caiu em toda a empresa", "Software de CRM com lentidão", "Monitor piscando",
        "Wi-Fi instável", "Sistema de ponto falhando", "Impressora não imprime", "Problema intermitente no acesso remoto",
        "Sistema de faturamento fora do ar", "Mouse não está funcionando corretamente", "Falha na atualização do software",
        "Servidor sem resposta", "Conexão com a VPN está muito lenta", "Erro ao abrir o aplicativo de gestão",
        "Falha no disco rígido do servidor"
    ]
    
    # Gerar 100 registros com datas aleatórias para aumentar a variabilidade dos dados
    dados = []
    for _ in range(100):
        nome = random.choice(nomes)
        categoria = random.choice(categorias)
        prioridade = random.choice(prioridades)
        descricao = random.choice(descricoes)
        data_abertura = gerar_data_aleatoria()
        dados.append((nome, categoria, prioridade, descricao, data_abertura))

    # Inserir dados no banco de dados
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO chamados (nome, categoria, prioridade, descricao, data_abertura)
            VALUES (?, ?, ?, ?, ?)
        """, dados)
        conn.commit()

def gerar_data_aleatoria():
    """Gera uma data aleatória dentro de um intervalo de 2 anos"""
    # Data de 2 anos atrás até hoje
    data_inicio = datetime.now() - timedelta(days=730)
    delta = timedelta(days=random.randint(0, 730))
    return (data_inicio + delta).strftime("%Y-%m-%d %H:%M:%S")

# Inicializa o banco de dados com dados aleatórios
init_db()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/abrir_chamado", methods=["POST"])
def abrir_chamado():
    data = request.form
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chamados (nome, categoria, prioridade, descricao)
            VALUES (?, ?, ?, ?)
        """, (data["nome"], data["categoria"], data["prioridade"], data["descricao"]))
        conn.commit()
    return jsonify({"message": "Chamado aberto com sucesso!"})

@app.route("/listar_chamados")
def listar_chamados():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, nome, categoria, prioridade, status, data_abertura, data_fechamento
            FROM chamados
        """)
        chamados = cursor.fetchall()
    return jsonify(chamados)

@app.route("/fechar_chamado", methods=["POST"])
def fechar_chamado():
    chamado_id = request.form["id"]
    data_fechamento = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE chamados SET status = 'Fechado', data_fechamento = ?
            WHERE id = ?
        """, (data_fechamento, chamado_id))
        conn.commit()
    return jsonify({"message": "Chamado fechado com sucesso!"})

@app.route("/prever_demanda")
def prever_demanda():
    try:
        mes = request.args.get("mes")  # Mês no formato YYYY-MM
        categoria = request.args.get("categoria")  # Categoria (Rede, Software, Hardware)

        # Verificar se os parâmetros estão corretos
        if not mes or not categoria:
            return jsonify({"error": "Parâmetros 'mes' e 'categoria' são obrigatórios"}), 400
        
        # Recupera os dados históricos do banco de dados
        with sqlite3.connect(DB_PATH) as conn:
            query = """
                SELECT strftime('%Y-%m', data_abertura) AS ano_mes, categoria, COUNT(*)
                FROM chamados
                GROUP BY strftime('%Y-%m', data_abertura), categoria
                """
            df = pd.read_sql(query, conn)
        
        # Filtra os dados para a categoria selecionada
        df_categoria = df[df["categoria"] == categoria]

        # Se não houver dados para a categoria selecionada
        if df_categoria.empty:
            return jsonify({"error": f"Nenhum dado encontrado para a categoria '{categoria}'"}), 400
        
        # Verifica a quantidade de dados disponíveis para a categoria
        num_registros = len(df_categoria)
        
        # Se houver poucos dados, avisa que o modelo pode ser impreciso
        if num_registros < 5:
            return jsonify({"warning": "Dados históricos insuficientes para previsão precisa. Retornando a média dos chamados."}), 200

        # Converte o 'ano_mes' para datetime e extrai o mês como variável numérica
        df_categoria["mes"] = pd.to_datetime(df_categoria["ano_mes"], format="%Y-%m")
        
        # Prepara os dados para o modelo de ML
        df_categoria["ano_mes_int"] = df_categoria["mes"].apply(lambda x: x.year * 12 + x.month)  # Convertendo ano e mês para valor numérico
        X = df_categoria[["ano_mes_int"]]  # Usando ano e mês como variável independente
        y = df_categoria["COUNT(*)"]  # O número de chamados como variável dependente
        
        # Divisão dos dados em treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modelo RandomForest
        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        # Convertendo o mês solicitado para a variável numérica correspondente
        try:
            mes_int = datetime.strptime(mes, "%Y-%m")
            mes_int = mes_int.year * 12 + mes_int.month
        except ValueError:
            return jsonify({"error": "Formato do mês inválido. Use o formato YYYY-MM."}), 400

        # Se o mês solicitado não estiver presente nos dados históricos, usa o modelo para prever
        demanda_prevista = modelo.predict([[mes_int]])[0]

        return jsonify({"demanda_prevista": int(demanda_prevista)})

    except Exception as e:
        print("Erro ao prever demanda:", e)  # Log do erro no terminal
        return jsonify({"error": "Erro ao prever demanda", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
