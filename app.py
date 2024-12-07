from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import random
import io
import matplotlib.pyplot as plt

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
        excluir_dados_antigos()
        inserir_dados_exemplo()

def excluir_dados_antigos():
    """Exclui todos os dados antigos na tabela chamados"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chamados")
        conn.commit()

def inserir_dados_exemplo():
    """Insere dados de exemplo com maior variabilidade e sazonalidade"""
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
    
    meses_com_alta_demanda = [1, 6, 11]  # Janeiro, Junho, Novembro
    num_registros = 1000  # Aumentar para 1000 registros

    dados = []
    for _ in range(num_registros):
        nome = random.choice(nomes)
        categoria = random.choice(categorias)
        prioridade = random.choice(prioridades)
        descricao = random.choice(descricoes)
        data_abertura = gerar_data_aleatoria(meses_com_alta_demanda)
        dados.append((nome, categoria, prioridade, descricao, data_abertura))

    # Inserir dados no banco de dados
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO chamados (nome, categoria, prioridade, descricao, data_abertura)
            VALUES (?, ?, ?, ?, ?)
        """, dados)
        conn.commit()

def gerar_data_aleatoria(meses_com_alta_demanda):
    """Gera uma data aleatória dentro de um intervalo de 2 anos com sazonalidade"""
    data_inicio = datetime.now() - timedelta(days=730)
    delta_dias = random.randint(0, 730)
    data_aleatoria = data_inicio + timedelta(days=delta_dias)
    
    if data_aleatoria.month in meses_com_alta_demanda:
        multiplicador = random.choices([1, 2, 3], weights=[50, 30, 20])[0]  # Pesos para maior frequência
        data_aleatoria += timedelta(days=random.randint(0, multiplicador * 10))
    
    return data_aleatoria.strftime("%Y-%m-%d %H:%M:%S")

# Inicializa o banco de dados com dados aleatórios
init_db()

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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prever_demanda")
def prever_demanda():
    try:
        mes = request.args.get("mes")  # Mês no formato YYYY-MM
        categoria = request.args.get("categoria")  # Categoria (Rede, Software, Hardware)

        if not mes or not categoria:
            return jsonify({"error": "Parâmetros 'mes' e 'categoria' são obrigatórios"}), 400
        
        # Recupera os dados históricos do banco de dados
        with sqlite3.connect(DB_PATH) as conn:
            query = """
                SELECT strftime('%Y-%m', data_abertura) AS ano_mes, categoria, COUNT(*) as chamados
                FROM chamados
                GROUP BY strftime('%Y-%m', data_abertura), categoria
            """
            df = pd.read_sql(query, conn)
        
        # Filtra os dados para a categoria selecionada
        df_categoria = df[df["categoria"] == categoria]

        if len(df_categoria) < 5:
            return jsonify({"warning": "Dados históricos insuficientes para previsão precisa."}), 200

        # Prepara os dados para o modelo
        df_categoria["ano_mes"] = pd.to_datetime(df_categoria["ano_mes"])
        df_categoria["ano_mes_int"] = df_categoria["ano_mes"].apply(lambda x: x.year * 12 + x.month)
        X = df_categoria[["ano_mes_int"]]
        y = df_categoria["chamados"]

        # Divisão dos dados em treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modelo RandomForest
        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        # Convertendo o mês solicitado para a variável numérica correspondente
        mes_int = datetime.strptime(mes, "%Y-%m")
        mes_int = mes_int.year * 12 + mes_int.month

        # Previsão
        demanda_prevista = modelo.predict([[mes_int]])[0]

        return jsonify({"demanda_prevista": int(demanda_prevista)})

    except Exception as e:
        print("Erro ao prever demanda:", e)
        return jsonify({"error": "Erro ao prever demanda", "details": str(e)}), 500

@app.route("/grafico_categoria/<categoria>")
def grafico_categoria(categoria):
    try:
        # Conectar ao banco de dados e carregar os dados
        with sqlite3.connect(DB_PATH) as conn:
            query = """
                SELECT strftime('%Y-%m', data_abertura) AS ano_mes, categoria, COUNT(*) as chamados
                FROM chamados
                GROUP BY strftime('%Y-%m', data_abertura), categoria
            """
            df = pd.read_sql(query, conn)

        if df.empty or categoria not in df["categoria"].unique():
            return jsonify({"error": f"Nenhum dado encontrado para a categoria '{categoria}'"}), 404

        # Filtrar e preparar os dados
        df["ano_mes"] = pd.to_datetime(df["ano_mes"], format='%Y-%m')
        df_categoria = df[df["categoria"] == categoria]

        # Geração do gráfico
        plt.figure(figsize=(10, 6))
        plt.plot(df_categoria["ano_mes"], df_categoria["chamados"], marker="o", label="Chamados")
        plt.title(f"Gráfico de Chamados - Categoria: {categoria}", fontsize=16)
        plt.xlabel("Mês", fontsize=14)
        plt.ylabel("Número de Chamados", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Salvar e retornar gráfico
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return send_file(img, mimetype='image/png')

    except Exception as e:
        print(f"Erro ao gerar gráfico: {e}")
        return jsonify({"error": "Erro ao gerar gráfico", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
