# ----------------------------------------------------------------------
# Parameters

FEAT = bert
#BERT = biaffine-dep-xlmr
BERT = bert-base-multilingual-cased
MODEL = --bert=$(BERT)
CONFIG = config.ini

GPU = 0

#BUCKETS = --buckets=48
#BATCH_SIZE = --batch-size=500
#MAX_SENT_LENGTH=--max-sent-length 140
#ATTN=--attention-layer=6

#----------------------------------------------------------------------
# Corpora

CORPUS_DIR = ../train-dev
CORPUS_TRAIN = $(CORPUS_DIR)/UD_$(RES2)/$(CORPUS)-ud-train.conllu
CORPUS_DEV = $(CORPUS_DIR)/UD_$(RES2)/$(CORPUS)-ud-dev.conllu
CORPUS_TEST = $(CORPUS_DIR)/UD_$(RES2)/$(CORPUS)-ud-test.conllu

BLIND_TEST=../test-blind/$(LANG).conllu
GOLD_TEST= ../test-gold/$(LANG).conllu

UD_TOOLS = ../tools
EVALB = python ../eval.py
EVAL18 = ../conll18_ud_eval.py
XUD_EVAL = ../iwpt21_xud_eval.py

ifeq ($(LANG), ar)
  CORPUS=ar_padt
  RES2=Arabic-PADT
  MODEL = --bert=asafaya/bert-large-arabic #TurkuNLP/wikibert-base-ar-cased
  BERT = asafaya
else ifeq ($(LANG), ba)
  CORPUS=ba
  RES2=Baltic
  #MODEL = --bert=TurkuNLP/wikibert-base-lv-cased
else ifeq ($(LANG), bg)
  CORPUS=bg_btb
  RES2=Bulgarian-BTB
  MODEL = --bert=DeepPavlov/bert-base-bg-cs-pl-ru-cased #TurkuNLP/wikibert-base-bg-cased #iarfmoose/roberta-base-bulgarian
  BERT = DeepPavlov
else ifeq ($(LANG), ca)
  CORPUS_DIR=../ud-treebanks-v2.6
  BLIND_TEST= $(CORPUS_TEST)
  GOLD_TEST= $(CORPUS_TEST)
  CORPUS=ca_ancora
  RES2=Catalan-AnCora
  BERT = mbert
else ifeq ($(LANG), cs) #dev PDT
  CORPUS=cs_czech
  RES2=Czech
  MODEL = --bert=DeepPavlov/bert-base-bg-cs-pl-ru-cased
  BERT = DeepPavlov
else ifeq ($(LANG), de)
  CORPUS_DIR=../ud-treebanks-v2.6
  BLIND_TEST= $(CORPUS_TEST)
  GOLD_TEST= $(CORPUS_TEST)
  CORPUS=de_hdt
  RES2=German-HDT
  MODEL = --bert=dbmdz/bert-base-german-uncased
  BERT = dbmdz-bert-base
else ifeq ($(LANG), en)
  CORPUS=en_english
  RES2=English
  MODEL = --bert=google/electra-base-discriminator
  BERT = electra-base
else ifeq ($(LANG), ptb)
 CORPUS_DIR=..
  CORPUS=en_ptb
  CORPUS_TRAIN = $(CORPUS_DIR)/SD_English_PTB/$(CORPUS)-sd-train.conllu
  CORPUS_DEV = $(CORPUS_DIR)/SD_English_PTB/$(CORPUS)-sd-dev.conllu
  BLIND_TEST = $(CORPUS_DIR)/SD_English_PTB/$(CORPUS)-sd-test.conllu
  GOLD_TEST = $(CORPUS_DIR)/SD_English_PTB/en_ptb-sd-test.conllu
  MODEL = --bert=google/electra-base-discriminator
  BERT = electra-base
else ifeq ($(LANG), es)
  CORPUS_DIR=../ud-treebanks-v2.6
  BLIND_TEST= $(CORPUS_TEST)
  GOLD_TEST= $(CORPUS_TEST)
  CORPUS=es_ancora
  RES2=Spanish-AnCora
  #MODEL = --bert=skimai/electra-small-spanish # TurkuNLP/wikibert-base-es-cased
  BERT = mbert
else ifeq ($(LANG), et) #dev EDT
  CORPUS=et_estonian
  RES2=Estonian
  #MODEL = --bert=TurkuNLP/wikibert-base-et-cased
  #BERT = mbert
else ifeq ($(LANG), fi)
  CORPUS=fi_tdt
  RES2=Finnish-TDT
  MODEL = --bert=TurkuNLP/bert-base-finnish-cased-v1
  #MODEL = --bert=TurkuNLP/wikibert-base-fi-cased
  BERT = TurkuNLP
  #ATTN=--attention-layer=8
else ifeq ($(LANG), fr)
  CORPUS=fr_sequoia
  RES2=French-Sequoia
  MODEL = --bert=dbmdz/bert-base-french-europeana-cased #camembert/camembert-large #camembert-base TurkuNLP/wikibert-base-fr-cased
  BERT = dbmdz-bert-base
  #BERT = camembert-large
else ifeq ($(LANG), it)
  CORPUS=it_isdt
  RES2=Italian-ISDT
  MODEL = --bert=dbmdz/electra-base-italian-xxl-cased-discriminator
  BERT = dbmdz-electra-xxl
else ifeq ($(LANG), ja)
  CORPUS_DIR=../ud-treebanks-v2.6
  BLIND_TEST= $(CORPUS_TEST)
  GOLD_TEST= $(CORPUS_TEST)
  CORPUS=ja_gsd
  RES2=Japanese-GSD
  MODEL = --bert=cl-tohoku/bert-base-japanese
  bert = cl-tohoku-bert
else ifeq ($(LANG), la)
  CORPUS_DIR=../ud-treebanks-v2.6
  BLIND_TEST= $(CORPUS_TEST)
  GOLD_TEST= $(CORPUS_TEST)
  CORPUS=la_ittb_llct
  RES2=Latin-ITTB-LLCT
  #MODEL = --bert=ponteineptique/latin-classical-small
  BERT = mbert
else ifeq ($(LANG), lt)
  CORPUS=lt_alksnis
  RES2=Lithuanian-ALKSNIS
  #MODEL = --bert=TurkuNLP/wikibert-base-lt-cased
  ATTN=--attention-layer=8
else ifeq ($(LANG), lv)
  CORPUS=lv_lvtb
  RES2=Latvian-LVTB
  #MODEL = --bert=TurkuNLP/wikibert-base-lv-cased
else ifeq ($(LANG), nl) #dev Alpino
  CORPUS=nl_dutch
  RES2=Dutch
  #MODEL = --bert=TurkuNLP/wikibert-base-nl-cased
  MODEL = --bert=wietsedv/bert-base-dutch-cased
  BERT = wietsedv
else ifeq ($(LANG), no)
  CORPUS_DIR=../ud-treebanks-v2.6
  BLIND_TEST= $(CORPUS_TEST)
  GOLD_TEST= $(CORPUS_TEST)
  CORPUS=no_nynorsk
  RES2=Norwegian-Nynorsk
  MODEL = --bert=TurkuNLP/wikibert-base-no-cased
  BERT = turkunlp
else ifeq ($(LANG), pl) #dev LFG
  CORPUS=pl_polish
  RES2=Polish
  MODEL = --bert=dkleczek/bert-base-polish-cased-v1 #DeepPavlov/bert-base-bg-cs-pl-ru-cased
  BERT = dkleczek
else ifeq ($(LANG), ro)
  CORPUS_DIR=../ud-treebanks-v2.6
  BLIND_TEST= $(CORPUS_TEST)
  GOLD_TEST= $(CORPUS_TEST)
  CORPUS=ro_rrt
  RES2=Romanian-RRT
  BERT = mbert
else ifeq ($(LANG), ru)
  CORPUS=ru_syntagrus
  RES2=Russian-SynTagRus
  MODEL = --bert=DeepPavlov/rubert-base-cased
  BERT = DeepPavlov
else ifeq ($(LANG), sk)
  CORPUS=sk_snk
  RES2=Slovak-SNK
  #MODEL = --bert=TurkuNLP/wikibert-base-sk-cased
else ifeq ($(LANG), sv)
  CORPUS=sv_talbanken
  RES2=Swedish-Talbanken
  MODEL = --bert=KB/bert-base-swedish-cased
  BERT = KB
else ifeq ($(LANG), ta)
  CORPUS=ta_ttb
  RES2=Tamil-TTB
  BLIND_TEST = $(CORPUS_DIR)/../test-udpipe/$(LANG).conllu
  #MODEL = --bert=monsoon-nlp/tamillion
else ifeq ($(LANG), uk)
  CORPUS=uk_iu
  RES2=Ukrainian-IU
  MODEL = --bert=dbmdz/electra-base-ukrainian-cased-discriminator #TurkuNLP/wikibert-base-uk-cased
  BERT = dbmdz-electra-base
  # nu=0.9
else ifeq ($(LANG), zh)
  CORPUS=zh_ctb7
  CORPUS_TRAIN = $(CORPUS_DIR)/CoNLL09/$(CORPUS)-train.conllu
  CORPUS_DEV = $(CORPUS_DIR)/CoNLL09/$(CORPUS)-dev.conllu
  BLIND_TEST = $(CORPUS_DIR)/CoNLL09/$(CORPUS)-test.conllu
  GOLD_TEST = $(CORPUS_DIR)/CoNLL09/$(CORPUS)-test.conllu
  MODEL = --bert=hfl/chinese-electra-base-discriminator # bert-base-chinese # hfl/chinese-electra-large-discriminator
  BERT = hfl-eletcra-base
else
  CORPUS_TRAIN= data/CoNLL2009-ST-English-train.conll
  CORPUS_DEV  = data/CoNLL2009-ST-English-development.conll
  CORPUS_TEST = data/CoNLL2009-ST-English-test-wsj.conll
endif

#----------------------------------------------------------------------
# Targets

.PRECIOUS: exp/$(CORPUS).$(BERT)$(VER)/model

TARGET=exp/$(CORPUS).$(BERT)$(VER)

# relate LANG to CORPUS
exp/$(LANG)$(VER)%: $(TARGET)%
	@

$(TARGET)/model:
	CUDA_VISIBLE_DEVICES=$(GPU) python -u -m supar.cmds.biaffine_dep train -d=$(GPU) -b -p=$@ \
	   -c=$(CONFIG) $(MODEL) $(ATTN) \
	   --train=$(CORPUS_TRAIN) $(MAX_SENT_LENGTH) $(BATCH_SIZE) $(BUCKETS) \
	   --dev=$(CORPUS_DEV) --feat=$(FEAT) --encoder=bert --punct

# relate LANG to CORPUS
exp.edp/$(LANG)$(VER)%: $(TARGET).edp%
	@

$(TARGET).edp/model:
	CUDA_VISIBLE_DEVICES=$(GPU) python -u -m supar.cmds.biaffine_edp train -d=$(GPU) -b -p=$@ \
	   -c=$(CONFIG) $(MODEL) $(ATTN) \
	   --train=$(CORPUS_TRAIN) $(MAX_SENT_LENGTH) $(BATCH_SIZE) $(BUCKETS) \
	   --dev=$(CORPUS_DEV) --feat=$(FEAT) --encoder=bert --punct

exp.sdp/$(LANG)$(VER)%: $(TARGET).sdp%
	@
$(TARGET).sdp/model:
	CUDA_VISIBLE_DEVICES=$(GPU) python -u -m supar.cmds.biaffine_sdp train -d=$(GPU) -b -p=$@ \
	   -c=$(CONFIG) $(MODEL) $(ATTN) \
	   --train=$(CORPUS_TRAIN) $(MAX_SENT_LENGTH) $(BATCH_SIZE) $(BUCKETS) \
	   --dev=$(CORPUS_DEV) --feat=$(FEAT) --encoder=bert

../submission/$(LANG).conllu: $(TARGET)/model
	CUDA_VISIBLE_DEVICES=$(GPU) python -u -m supar.cmds.biaffine_dep predict -d=$(GPU) -p=$< --tree \
	   --data=$(BLIND_TEST) \
	   --pred=$@
	python $(UD_TOOLS)/fix-root.py $@

$(TARGET).dev.conllu: $(TARGET)/model
	CUDA_VISIBLE_DEVICES=$(GPU) python -u -m supar.cmds.biaffine_dep predict -d=$(GPU) -p=$< --tree \
	   --data=$(CORPUS_DEV) \
	   --pred=$@
#	python $(CORPUS_DIR)/../fix-root.py $@

LANGS=ar bg cs en et fi fr it lt lv nl pl ru sk sv ta uk 
XLMR_LANGS=bg cs en fr it nl ru
UD_LANGS=ca de es ja no ro

submission$(REL):
	for l in $(LANGS); do \
	    $(MAKE) -s GPU=$(GPU) LANG=$$l FEAT=$(FEAT) VER=$(VER) ../submission$(REL)/$$l.conllu &>> ../submission$(REL)/$${l}$(VER).make; \
	done

all:
	for l in $(LANGS); do \
	    $(MAKE) -s GPU=$(GPU) LANG=$$l FEAT=$(FEAT) VER=$(VER) exp/$${l}$(VER).test.eval18 &>> exp/$${l}$(VER).test.make; \
	done

all-ud:
	for l in $(UD_LANGS); do \
	    $(MAKE) -s GPU=$(GPU) LANG=$$l FEAT=$(FEAT) VER=$(VER) exp/$${l}$(VER).test.eval18 &>> exp/$${l}$(VER).test.make; \
	done

dev:
	for l in $(LANGS); do \
	    $(MAKE) -s GPU=$(GPU) LANG=$$l FEAT=$(FEAT) VER=$(VER) exp/$${l}$(VER).dev.conllu &>> exp/$${l}$(VER).dev.make; \
	done

eval18:
	for l in $(LANGS); do \
	    $(MAKE) -s GPU=$(GPU) LANG=$$l FEAT=$(FEAT) VER=$(VER) exp/$${l}$(VER).test.eval18 &>> exp/$${l}$(VER).test.make; \
	done

train:
	for l in $(LANGS); do \
	    ${MAKE} -s GPU=$(GPU) LANG=$$l exp/$$l$(VER)/model &>> exp/$${l}$(VER).train.make; \
	done

# ----------------------------------------------------------------------
# Evaluation

$(TARGET).test.nen.conllu: $(TARGET).test.conllu
	perl $(UD_TOOLS)/enhanced_collapse_empty_nodes.pl $< > $(TARGET).test.nen.conllu

$(TARGET).test.eval: $(TARGET).test.nen.conllu
	python $(XUD_EVAL) -v $(UD_TOOLS)/../test-gold/$(LANG).nen.conllu $(TARGET).test.nen.conllu > $@

$(TARGET).test.evalb: $(TARGET).test.eval
	$(EVALB) -g $(GOLD_TEST) -s $@ --evalb

$(TARGET).test.eval18: $(TARGET).test.conllu
	$(EVAL18) $(GOLD_TEST) $< > $@

$(TARGET).dev.eval18: $(TARGET).dev.conllu
	$(EVAL18) $(CORPUS_DEV) $< > $@

evaluate:
	for l in $(LANGS); do \
	   $(MAKE) -s GPU=$(GPU) LANG=$$l exp/$$l.$(BERT).test.evalb &>> exp/$$l.$(BERT).test.make; \
	done

exp/test.eval: evaluate
	( cd exp; python ../eval-summary.py > $(notdir $@) )

# ----------------------------------------------------------------------
# Run tests

lint:
	flake8 parser --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

test:
	pytest -s tests
