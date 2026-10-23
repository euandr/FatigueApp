## Migração das comunicações para FastAPI

Estou desenvolvendo um sistema de detecção de fadiga com React no frontend, Python no backend e Supabase para autenticação e banco de dados.

O arquivo abaixo contém a documentação da estrutura atual do banco de dados e deve ser consultado antes de iniciar as alterações:

`<span>C:\Users\andre\OneDrive\Documentos\FatigueApp\notas\DATABASE_atualizado.md</span>`

### Objetivo

Atualmente, parte das comunicações com o Supabase é realizada diretamente pelo frontend enem esta completo. Quero transferir as operações de banco de dados para o backend, utilizando **FastAPI**.

Também quero migrar o servidor WebSocket atual, implementado com a biblioteca `<span>websockets</span>`, para o **WebSocket do FastAPI**.

### O que deve ser feito

1. **Configurar o FastAPI**
   * Instalar as dependências necessárias.
   * atualizar o requirements.txt
   * Organizar o backend em arquivos e módulos com responsabilidades bem definidas.
   * Criar endpoints REST para as operações necessárias com o banco de dados.
2. **Integrar o backend ao Supabase**
   * Transferir para o FastAPI as operações de leitura e escrita atualmente realizadas pelo frontend.
   * Utilizar a estrutura de banco documentada no arquivo informado.
   * Manter a autenticação existente com Supabase Auth.
   * Validar no backend o token JWT dos usuários autenticados.
   * Não expor credenciais privadas do Supabase no frontend.
3. **Migrar o WebSocket**
   * Substituir o servidor WebSocket atual pelo WebSocket do FastAPI.
   * Preservar o funcionamento da comunicação em tempo real.
   * Manter o envio de frames, métricas e alertas conforme o funcionamento atual.
   * Preservar, sempre que possível, o formato das mensagens já utilizado pelo frontend.
4. **Ajustar o frontend**
   * Substituir as chamadas diretas ao banco de dados pelas chamadas aos endpoints do FastAPI.
   * Atualizar a conexão WebSocket para utilizar a nova rota.
   * Manter o login com Supabase Auth.
   * Preservar as páginas, componentes, estilos e funcionalidades existentes.

### Restrições importantes

* **Não modificar a lógica de detecção de fadiga.** Os algoritmos e os cálculos existentes já estão implementados e funcionando.
* Não alterar os limiares, as métricas ou as condições utilizadas para detectar fadiga.
* Não refazer o frontend nem modificar sua aparência.
* Alterar apenas o necessário para integrar a nova API e o WebSocket.
* Não modificar a estrutura do banco de dados sem necessidade ou autorização.
* Não adicionar funcionalidades que não foram solicitadas.
* Evitar abstrações desnecessárias, arquiteturas excessivamente complexas e dependências que não sejam essenciais.

### Organização do código

criar uma pasta `<span>app</span>`dentro de `<span>backend</span>`para organizar a aplicação FastAPI, não meixa no que esta na pasta /detector_fadiga, ela vai ficar como codigo inicial caso eu queria fazer mudancas manualmente vou usar ela.

use portugues

Quero uma implementação **simples, organizada e fácil de manter**, utilizando nomes claros e comentários em português quando forem úteis.

Antes de implementar, analise a estrutura atual do projeto para aproveitar o código existente e evitar alterações desnecessárias.

Após a implementação, gere um arquivo em /notas que explique quais arquivos foram criados ou modificados, quais dependências foram instaladas e como executar e testar o sistema.

**Prioridade:** realizar a migração das comunicações para o FastAPI sem comprometer o funcionamento atual do sistema de detecção de fadiga.


se houver alguma duvida me pergunte no formato de respstar de marcar, durante a eecução para que nao interrompa ate terminar a implementação.
