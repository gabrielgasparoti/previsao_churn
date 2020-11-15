##############################################################
## REGRESSAO LOGISTICA BINARIA
#############################################################

#OBJETIVO: PREVER A PROBABILIDADE DE UM CONSUMIDOR SER CLASSIFICADO
#COMO CHURN COM BASE EM 20 ATRIBUTOS ATRAVES DE UM MODEO LOGISTICO BINARIO

# Fonte de dados : https://www.kaggle.com/blastchar/telco-customer-churn
# Data Mining previamente realizado com python pandas

#IMPORTANDO OS DADOS E AJUSTANDO O INDICE
library(tidyverse)
dados <- read.csv("dados_consumidor.csv")
row.names(dados) <- dados$ID
dados$ID <- NULL


#VISUALIZANDO O TIPO DE VARIAVEL
glimpse(dados)

#AJUSTANDO VARIAVEIS QUALITATIVAS
dados$GENERO <- as.factor(dados$GENERO)
dados$IDOSO <- as.factor(dados$IDOSO)
dados$CASADO <- as.factor(dados$CASADO)
dados$FILHOS <- as.factor(dados$FILHOS)
dados$SERVICO_TELEFONICO <- as.factor(dados$SERVICO_TELEFONICO)
dados$LINHAS_MULTIPLAS <- as.factor(dados$LINHAS_MULTIPLAS)
dados$PROVEDOR_INTERNET <- as.factor(dados$PROVEDOR_INTERNET)
dados$SEGURANCA_ONLINE <- as.factor(dados$SEGURANCA_ONLINE)
dados$BACKUP_ONLINE <- as.factor(dados$BACKUP_ONLINE)
dados$PROTECAO_DISPOSITIVO <- as.factor(dados$PROTECAO_DISPOSITIVO)
dados$SUPORTE_TECNICO <- as.factor(dados$SUPORTE_TECNICO)
dados$STREAMING_TV <- as.factor(dados$STREAMING_TV)
dados$STREAMING_MOVIES <- as.factor(dados$STREAMING_MOVIES)
dados$TEMPO_VIGENCIA_CONTRATO <- as.factor(dados$TEMPO_VIGENCIA_CONTRATO)
dados$FATURAMENTO_SEM_PAPEL <- as.factor(dados$FATURAMENTO_SEM_PAPEL)
dados$FORMA_DE_PAGAMENTO <- as.factor(dados$FORMA_DE_PAGAMENTO)
dados$CHURN <- as.factor(dados$CHURN)


#ANALISE DE FREQUENCIA ABSOLUTA DAS VARIVEIS QUALITATIVAS
#E ANALISE DE MAXIMOS E MINIMOS DE VARIAVEIS QUANTITATIVAS
summary(dados)

#SOBRE AS OBSERVACOES:
#16% SAO IDOSOS
#48% SAO CASADOS
#30% TEM FILHOS
#EM MEDIA OS CLIENTES FICAM 32 MESES COM CONTRATO ATIVO
#90% ASSINA O SERVICO TELEFONICO
#A MAIORIA DOS QUE POSSUEM INTERNET O PROVEDOR E FIBRA OTICA
#E NAO POSSUEM SEGURANCA ONLINE NEM BACKUP ONLINE
#NEM PROTECAO DO DISPOSITIVO NEM SUPORTE TECNICO
#O TEMPO DE VIGENCIA DO CONTRATO MAIS FREQUENTE E O MES A MES
#A FORMA DE PAGAMENTO MAIS FREQUENTE E O CHEQUE ELETRONICO
#O VALOR MEDIO DA MENSALIDADE E DE 64,80
#27% SAO CHURN


#TRATAMENTO DE VARIVEIS DUMMIES
library(fastDummies)
dados_dummies <- dummy_columns(.data = dados,
                               select_columns = c('GENERO','IDOSO','CASADO','FILHOS',
                                                  'SERVICO_TELEFONICO','LINHAS_MULTIPLAS',
                                                  'PROVEDOR_INTERNET','SEGURANCA_ONLINE',
                                                  'BACKUP_ONLINE','PROTECAO_DISPOSITIVO',
                                                  'SUPORTE_TECNICO','STREAMING_TV','STREAMING_MOVIES',
                                                  'TEMPO_VIGENCIA_CONTRATO','FATURAMENTO_SEM_PAPEL',
                                                  'FORMA_DE_PAGAMENTO','CHURN'),
                               remove_selected_columns = T,
                               remove_most_frequent_dummy  = T)


#VISUALIZANDO A BASE DE DADOS
library(kableExtra)
head(dados_dummies) %>%
  kable()%>%
  kable_styling(bootstrap_options = "striped", font_size = 12)

#CONFIRMANDO SE O TIPO QUALI FOI EXTINTO DO DATASET
glimpse(dados_dummies)

#ESTIMANDO UM MODELO POR GLM
modelo <- glm(formula = CHURN_Yes ~ .,
              data = dados_dummies,
              family = "binomial")


#VISUALIZANDO OS PARAMETROS DO LOGITO E P-VALOR CORRESPONDENTE
library(jtools)
summary(modelo)
summ(modelo, confint = T, ci.width = 0.95)

#APLICANDO O PROCEDIMENTO STEPWISE PARA RETIRAR PARAMETROS 
#QUE NAO SAO ESTATISTICAMENTE SIGNIFICANTES A 95% DE CONFIANCA
modelo_step <- step(object = modelo, k = qchisq(p = 0.05, df = 1, lower.tail = FALSE))

#ANALISANDO NOVAMENTE OS PARAMETROS 
summary(modelo_step)

#alguns insights sobre os parametros com base na amostra rodada no modelo:
  ## quanto maior a mensalidade e/ou tempo com contrato ativo, se tem filhos,
  ## se nao possui internet e tempo de contrato for de um a dois anos
  ## menor a chance de ser churn

  ## quanto maior o valor mensal cobrado (mensalidade + outras taxas),
  ## se for idoso, se possui servico telefonico e se o provedor de internet
  ## for fibra otica, maior a chance de churn

#CONFIRMANDO SE TODOS OS PARAMETROS SAO ESTATISTICAMENTE
#SIGNIFICANTES A 95% DE CONFIANCA
summ(modelo_step, confint = T, ci.width = 0.95)

#EXTRACAO DO VALOR DO LOGLIK
#SOMATORIO DA PROBABILIDADE DE OCORRENCIA DO EVENTO
# OU DA PROBABILIDADE DE OCORRENCIA DO NAO EVENTO
# DE CADA OBSERVACAO
# NA COMPARACAO DE MODELOS, QUANTO MAIOR O LL MELHOR O MODELO
logLik(modelo_step)

#CRIANDO UM DATAFRAME COM O EVENTO REAL E AS PROBABILIDADES PREVISTAS PARA O EVENTO
probs <- as.data.frame(dados$CHURN)
probs$predict <- modelo_step$fitted.values
probs %>%
  kable()%>%
  kable_styling(bootstrap_options = "striped")


#AVALIACAO GERAL DO MODELO
#Matriz de confusao com cutoff de 0.4
library(caret)
cm <- confusionMatrix(table(predict(modelo_step, type = "response") >= 0.4, 
                            dados_dummies$CHURN_Yes == 1)[2:1, 2:1]) 

cm$table
cm$overall[1] #ACURACIA DE 79% (percentual de acerto geral)
cm$byClass[1] #SENSITIVIDADE 67%  (dos que sao evento, quantos o modelo preve que sao evento)
cm$byClass[2] #ESPECIFICIDADE 83% (dos que nao sao evento, quantos o modelo preve que nao sao evento)
cm$byClass[3] #PRECISAO 59% (daqueles que foram previstos como evento, quantos realmente sao)


#CONSTRUCAO E VISUALIZACAO DA CURVA ROC
library(pROC)
library(plotly)

roc <- roc(response = dados_dummies$CHURN_Yes,
           predictor = modelo_step$fitted.values)

ggplotly(
  ggroc(roc, color = "darkorchid", size=0.9)+
    geom_segment(aes(x = 1, xend = 0, y = 0, yend=1), color = 'orange')+
    labs(x = '1 - Especificidade', y = 'Sensitividade', 
         title = paste("AUC:", round(roc$auc,2), "|",
                       "GINI:", round((roc$auc-0.5)/0.5,2) ))+
    theme_bw()
)

#VISUALIZACAO EM 2D DAS VARIAVEIS METRICAS E AS PROBABILIDADES
probs$MESES_ATIVO <- dados_dummies$MESES_ATIVO
probs$MENSALIDADE <- dados_dummies$MENSALIDADE


#PROBABILIDADE DE CHURN EM FUNÇÃO DO TEMPO ATIVO EM MESES
# Metodo de suavizacao "loess"
# (quanto maior o período ativo, menor a chance de churn)
ggplotly(
  ggplot(data = probs,
         (aes(x = MESES_ATIVO, y= predict)))+
    geom_smooth(method = "loess", se = T)+
    labs( x = 'Meses com contrato ativo',
          y = 'Probabilidade de Churn',
          title = 'Probabilidade de Churn em função do tempo')
)


#PROBABILIDADE DE CHURN EM FUNCAO DO VALOR DA MENSALIDADE
# Metodo de suavizacao "loess"
# (suavizando a curva, no intervalo de 0 a 90, quanto maior a mensalidade, maior a chance do evento)
# (no intervalo acima de 90, a chance de ocorrer o evento diminui quando a mensalidade aumenta)
ggplotly(
  ggplot(data = probs,
         (aes(x = MENSALIDADE, y= predict)))+
    geom_smooth(method = "loess", se = T)+
    labs( x = "Valor da Mensalidade", y = "Probabilidade de Churn",
          title = "Probabilidade de Churn X Valor Mensal")
)
