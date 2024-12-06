# CMOS Circuit Optimizer with PSO and PySpice

Este projeto utiliza a biblioteca PySpice para simular circuitos CMOS e aplica o algoritmo Particle Swarm Optimization (PSO) para otimizar parâmetros de projeto.

## Conteúdo do Repositório

- `cmos_inverter.py`: Script para simulação de um inversor CMOS utilizando PySpice.
- `diferential_pair.py`: Script para simulação de um par diferencial CMOS utilizando PySpice.

## Requisitos

- Python 3.x
- [PySpice](https://pyspice.fabrice-salvaire.fr/)

## Instalação

1. Clone este repositório:

   ```bash
   git clone https://github.com/Klusmo/cmos_ci_pso_optimizer_with_pyspice.git
   ```

2. Navegue até o diretório do projeto:

   ```bash
   cd cmos_ci_pso_optimizer_with_pyspice
   ```

3. Instale as dependências necessárias:

   ```bash
   pip install -r requirements.txt
   ```

   *Nota: Certifique-se de que o PySpice está instalado corretamente conforme as instruções oficiais.*

## Uso

1. Para simular o inversor CMOS:

   ```bash
   python cmos_inverter.py
   ```

2. Para simular o par diferencial CMOS:

   ```bash
   python diferential_pair.py
   ```

Os resultados das simulações serão exibidos no console ou salvos em arquivos, conforme implementado nos scripts.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

 