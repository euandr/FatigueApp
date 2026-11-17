# pre-detecção

- [x]  depois de dar start no video, ele deve continuar aparecendo enquanto eu nao fechar a conexão no botao. Caso eu troque da tela de detalhamento da camera para a tela de monitoramento onde aparece todas as cameras, o video de exibição deve continuar aparecendo na cardCamera de forma miniaturizada.
     -  por enquanto quando fechar a pagina por completo como fechar o navegador vai perder a conexão de tudo, mas isso sera ajustado depois.

- [x] atualizar os status de monitoramento das duas telas.

---

- [x] agora é passsar a detecção para cada camera ativa. Por enquanto apenas detectar os estados e emitir alertas em tempo real.
-  a seguir trabalharei em guardar esses dados no BD, configurando api e frontend.


## ajustes
vamos fazer ajsutes antes de algo mais, agora. 

- em Detalhes não esta aparecendo nada.
- eu atualizei o value de protocolo de conecção, mas nao atulizou no banco e ja fui em editar e salvei (me refiro ao modal)
- precisamos ajustar o tempo de monitoramento
     - campo sessão iniciada
     - campo duracao
     - em resumo da sessão ajustar o tempo de monitoramento. Enquanto nao for iniciada a nova sessão ele mostra o tempo da ultima sessão