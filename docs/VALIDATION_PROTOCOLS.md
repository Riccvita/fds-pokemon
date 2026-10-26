# Validazione: protocolli separati e riproducibili

Questa guida descrive la **copia locale modificata**, non una versione che si
presume già pubblicata nel repository originale. Non sono stati modificati i
parametri di LightGBM, XGBoost o CatBoost, né i pesi 100%, 50/50 e 45/45/10.
La metrica resta l'accuracy; una probabilità **strettamente maggiore di 0,5**
predice 1, altrimenti 0.

I dati originali non sono presenti nel workspace usato per questa modifica.
Le verifiche sintetiche controllano il codice, non stimano le prestazioni sulle
10.000 battaglie. Nessun nuovo risultato di competizione è dichiarato.

## 1. Salvare e interpretare ogni previsione

**Domanda:** il guadagno deriva da poche correzioni stabili oppure da molti
scambi fra errori corretti e introdotti?

```bash
python run.py --data /percorso/dati --validation-only \
  --cv-out validation_runs/original
```

`validation_predictions.csv` conserva l'ordine originale del training e contiene:

- `row_id`: posizione originale, a partire da zero; identifica la riga anche
  quando l'ID ufficiale è duplicato.
- `battle_id`, `fold` (0–4), `player_won`.
- Una probabilità di vittoria per `lgb`, `xgb`, `cat`, `sub1_lgbm`,
  `sub2_lgb_xgb`, `sub3_lgb_xgb_cat` e `majority`.
- Per ogni nome, `<nome>__prediction` e `<nome>__correct` (0 oppure 1).
- Componenti di duplicazione, gruppi di composizione P1, impronta delle
  osservazioni e indicatori della presenza della squadra nell'addestramento.

Il riferimento `majority` predice la classe più frequente nel solo addestramento
esterno; i pareggi scelgono 0 e le sue probabilità sono 0 o 1. Ogni probabilità
valutata proviene da un modello che non ha addestrato sulla relativa battaglia.

`validation_report.json` include accuracy, corretti/sbagliati, conteggi e accuracy
per gruppo, correzioni, peggioramenti, saldo e differenza di accuracy rispetto a
LightGBM (`sub1_lgbm`). `corrected - regressed = net_correct`; il saldo diviso
per il numero di battaglie è la differenza di accuracy. Le accuracy dei gruppi
vanno pesate per la loro numerosità, soprattutto con raggruppamenti.

Il report registra seed, regola di decisione, modelli/pesi, distribuzione delle
classi, schema delle caratteristiche e impronta dello stato delle trasformazioni
per gruppo, dipendenze installate, Python/piattaforma, commit, stato Git, hash
**dei file effettivamente eseguiti**, hash dei dati originali e delle assegnazioni
ai gruppi. Il commit da solo non identifica modifiche non ancora committate.

Per confrontare una nuova esecuzione con una precedente:

```bash
python run.py --data /percorso/dati --validation-only \
  --cv-out validation_runs/nuova \
  --reference-oof validation_runs/original/validation_predictions.csv
```

Il confronto riallinea per ID e rifiuta etichette, osservazioni (quando presenti)
o gruppi diversi. Con ID duplicati usa anche la posizione originale: mantenere
lo stesso ordine del file sorgente. Sono ancora leggibili i precedenti CSV di
sole probabilità, incluso `cv_oof.csv.gz`, ma non vengono più generati i vecchi
nomi dei file. I riepiloghi storici senza previsioni individuali non bastano.

`cross_validate(data)` restituisce ancora un dizionario nome→accuracy, compreso
`majority` già aggiunto localmente. `return_details=True` restituisce report e
DataFrame. `--out` resta dedicato alle submission, `--cv-out` ai diagnostici;
`--validation-only` evita il successivo addestramento delle submission. Per
riprodurre il flusso precedente con entrambi: `--cv --out submissions`.

## 2. Ripetere cambiando solo la divisione esterna

**Domanda:** quanto dipende il confronto dalla particolare divisione casuale?

```bash
python run.py --data /percorso/dati --validation-only --repeat-cv \
  --cv-out validation_runs/repeated
```

I seed esterni prefissati sono **42, 123, 2026**. Il seed dei modelli resta **42**,
quello del target encoding **0**. Le cartelle `seed_42`, `seed_123`, `seed_2026`
contengono risultati separati. `repeated_validation_report.json` riporta tutti i
punteggi, differenze rispetto a LightGBM, media, minimo, massimo e deviazione
standard descrittiva. Nessun seed viene scelto come vincitore.

Le stesse battaglie ricompaiono: non sono 30.000 campioni indipendenti. Non si
producono intervalli di confidenza basati su tale falsa numerosità. Un vantaggio
con segno variabile fra ripetizioni indica sensibilità alla divisione.

La modalità singola usa sempre seed 42 salvo `--outer-seed`. I seed dei modelli
e dell'encoder sono parametri distinti nell'API, ma la CLI li mantiene fissi in
questa fase. I modelli all'interno di ciascuna ripetizione condividono i gruppi.

## 3. Duplicati, ricorrenze e generalizzazione a composizioni diverse

**Domande:** ci sono copie della stessa osservazione su entrambi i lati della
validazione? Il modello funziona anche su composizioni P1 assenti dall'addestramento?

Per il controllo senza addestrare:

```bash
python run.py --data /percorso/dati --audit-only \
  --cv-out validation_runs/audit
```

`data_audit.json` distingue ID duplicati, osservazioni identiche, eventuali
etichette in conflitto, composizioni ripetute in partite differenti, frequenze
delle composizioni e corrispondenze train/test. L'audit può funzionare anche
senza `test.jsonl`; gli esperimenti completi richiedono i tre file ufficiali.

Il criterio conservativo collega righe con **stesso ID oppure osservazioni
identiche**, escludendo soltanto `battle_id` e `player_won` dall'impronta. L'ordine
delle chiavi JSON non conta; quello delle liste e la rappresentazione numerica
restano significativi. Si considera la chiusura transitiva dei collegamenti.
Non è un rilevatore di quasi-duplicati, né la prova che due log uguali siano
necessariamente la stessa partita: i casi sospetti vanno ispezionati.
**Nessuna riga viene eliminata o rietichettata.**

Se esistono duplicati, i loro componenti restano nello stesso gruppo esterno
**e** nelle divisioni interne del target encoding e della selezione. Si usa
`StratifiedGroupKFold` cercando l'equilibrio delle classi. Il report conserva le
assegnazioni casuali originali come diagnostico, quantifica i componenti che
le attraversavano e quante assegnazioni cambiano. Non si addestrano risultati
con separazioni note di duplicati. In assenza di duplicati il protocollo casuale
rimane esattamente `StratifiedKFold(5, shuffle=True, random_state=42)`.

Nella validazione casuale, `composition_subgroups` separa composizioni P1 già
presenti come P1 nell'addestramento e composizioni assenti, con numerosità,
corretti/sbagliati e accuracy (anche per gruppo). Un sottogruppo vuoto ha accuracy
`null`, non zero.

Per escludere intere composizioni P1:

```bash
python run.py --data /percorso/dati --validation-only --group-by-p1 \
  --cv-out validation_runs/grouped
```

La composizione è l'insieme non ordinato delle specie, come nel target encoding
esistente; non distingue ordine, strumenti o mosse. Si uniscono anche i legami
fra duplicati: un ID anomalo può quindi collegare composizioni diverse. Gli
stessi componenti proteggono le divisioni interne dell'encoder. Se i gruppi
sono insufficienti, la procedura si ferma esplicitamente, senza ripiegare su
una divisione che ne viola i confini. L'encoder interno può ridurre i suoi
cinque gruppi al numero di gruppi disponibili (minimo due).

Una squadra esclusa come P1 può comparire come P2 completamente rivelata in
un'altra partita di addestramento e contribuire al target encoding. Il CSV e
il report misurano questa sovrapposizione, anche nel sottogruppo assente come P1.
La procedura **non la rimuove**: parla di composizioni escluse come P1, non di
squadre mai viste. Anche l'assenza da entrambi i lati *osservati* non garantisce
l'assenza fra le squadre P2 solo parzialmente rivelate, né dal catalogo globale.
Un calo rispetto alla validazione casuale descrive una diversa domanda di
generalizzazione; non è automaticamente una stima migliore della distribuzione
del test della competizione.

## 4. Confrontare due condizioni informative

**Domanda:** quanto incidono i cataloghi ricavati anche dagli ingressi delle
battaglie da valutare?

```bash
python run.py --data /percorso/dati --validation-only --preparation compare \
  --cv-out validation_runs/preparation
```

`global/` riproduce la preparazione attuale: vocabolari di mosse/specie e colonne
HP ricavati dall'intero training, dizionario delle statistiche di specie e
catalogo/frequenze delle squadre ricavati da training e test senza etichette.

`fold/` ricostruisce quegli stessi oggetti usando solo la porzione di
addestramento del gruppo corrente. Applica poi lo schema fissato alla
validazione, senza passarne le etichette al preparatore. Le soglie di frequenza
esistenti (100/150) restano uguali; il numero di colonne può quindi diminuire.
La tabella ufficiale dei tipi è condivisa. Le osservazioni direttamente presenti
in una battaglia, comprese le sue statistiche iniziali, rimangono utilizzabili.

Specie/mosse fuori vocabolario non creano nuove colonne. Per statistiche di
specie non presenti nel dizionario si usano i NaN/default già previsti dalle
caratteristiche; i confronti di tipo basati sul dizionario omettono gli elementi
sconosciuti. Se P2 mostra specie esterne al catalogo non ci sono squadre candidate:
il relativo target encoding resta neutro (0,5 e zero partite). Il catalogo non
viene aggiornato osservando la validazione.

`preparation_comparison.json` confronta i due risultati battaglia per battaglia,
con correzioni, peggioramenti e differenza di accuracy **fold meno global**.
Gruppi, modelli e parametri sono identici. Si può aggiungere `--group-by-p1`
per ripetere questo confronto sotto il protocollo per composizione; eseguirlo
come esperimento distinto. `--preparation fold` esegue soltanto il secondo caso.

L'uso di ingressi senza etichette non implica automaticamente fuga delle
risposte: questi sono protocolli con informazioni disponibili diverse.
L'ammissibilità del catalogo transduttivo va verificata nel regolamento della
competizione; questo lavoro non afferma di averne verificato l'autorizzazione.

## 5. Separare la scelta delle alternative dalla loro valutazione

**Domanda:** quale prestazione ottiene la procedura che sceglie una combinazione
senza usare gli esiti del gruppo esterno?

Prima scegliere esplicitamente il protocollo informativo dei passi precedenti.
Per confrontare la sola aggiunta della selezione, mantenere quel protocollo:

```bash
# Stessa preparazione globale del riferimento, aggiungendo solo la selezione.
python run.py --data /percorso/dati --validation-only --nested-cv \
  --cv-out validation_runs/nested_global

# Se il riferimento di interesse è già fold-local, usare invece:
python run.py --data /percorso/dati --validation-only --preparation fold --nested-cv \
  --cv-out validation_runs/nested_fold
```

I finalisti predefiniti sono **solo le tre combinazioni già presenti**. Si può
restringere la lista con `--finalists sub1_lgbm sub3_lgb_xgb_cat`. Non si cercano
nuovi pesi, caratteristiche o iperparametri. Per ogni gruppo esterno:

1. Tre gruppi interni (seed di selezione fisso 17) producono previsioni OOF solo
   sulle battaglie di addestramento esterno.
2. L'accuracy interna seleziona il finalista; un pareggio sceglie il primo
   nell'ordine dichiarato.
3. Si riaddestrano i modelli sull'addestramento esterno e si applica la scelta
   già fissata alla validazione esterna.

`nested_selected` misura l'intera procedura di selezione. I risultati interni,
i gruppi e le previsioni sono salvati in `selection/outer_0/` … `outer_4/`;
il report principale registra il finalista scelto per ogni gruppo. Non usare
i risultati esterni dei singoli finalisti per effettuare un'ulteriore scelta e
presentarla poi come indipendentemente valutata.

Nella preparazione `fold` anche vocabolari e cataloghi vengono riappresi sul
solo addestramento di ogni divisione interna e ricostruiti al riaddestramento
esterno. Nella preparazione `global` il catalogo senza etichette resta quello
globale prefissato: l'indipendenza riguarda le etichette, non gli ingressi.
Il target encoding viene sempre ricalcolato esclusivamente sulle etichette
ammesse, con un ulteriore cross-fitting delle righe di addestramento. In modalità
per squadra anche i gruppi interni di selezione e di encoding rispettano i
componenti di composizione.

Il cross-fitting dell'encoder, già esistente, **non equivaleva** alla separazione
fra selezione delle alternative e valutazione. Inoltre i numerosi esperimenti
storici hanno già influenzato la lista di finalisti: cambiare suddivisione o
aggiungere l'annidamento non cancella quella storia. Questa verifica riduce il
riutilizzo diretto delle etichette esterne nella nuova procedura, non crea un
nuovo test incontaminato.

La modalità è opzionale: con tre gruppi interni richiede circa quattro volte
gli addestramenti della validazione semplice. Il notebook la lascia disattivata.
Non viene avviata alcuna ricerca estesa e non si usa la leaderboard.

## Eseguire la propria copia nel notebook

Il notebook ora richiede una scelta esplicita della sorgente:

- `SOURCE_MODE = "local"`: `LOCAL_SOURCE_DIR` indica la cartella modificata,
  anche senza `.git`. Su Kaggle caricarla come Dataset; in locale si può usare
  `FDS_POKEMON_SOURCE=/percorso/fds-pokemon`. Non avviene alcun pull.
- `SOURCE_MODE = "fork"`: impostare URL del proprio fork e revisione, idealmente
  un commit preciso. Non servono permessi di push sul repository originale.
  Il notebook rifiuta di aggiornare un checkout sporco o con un altro remote.

Per includere **anche i file locali nuovi e non committati**, dalla radice:

```bash
tar -czf /tmp/fds-pokemon-source.tar.gz --exclude='__pycache__' \
  --transform='s,^,fds-pokemon/,' \
  pokemon notebooks tests docs run.py strength_experiment.py \
  action_experiment.py transition_experiment.py research_70.py \
  requirements.txt requirements-kaggle.txt README.md
```

Estrarre l'archivio e caricare la cartella `fds-pokemon` nel Dataset scelto.
Adattare `LOCAL_SOURCE_DIR` al percorso effettivo mostrato da Kaggle: il notebook
non presume che i file appena modificati siano già sul repository originale.
Non usare `git archive HEAD` per questo passaggio: escluderebbe le modifiche
locali. Se la consegna finale impone un clone pubblico, configurare il proprio
fork e verificare il requisito prima della consegna.

Il notebook stampa directory, commit e `pokemon.main.__file__`; salva le impronte
della sorgente effettiva nel manifest. Riavviare il kernel cambiando copia o
versione delle dipendenze. La cella opzionale permette un solo protocollo alla
volta; i diagnostici sono in `/kaggle/working/validation`, separati dalle
submission. Nessun protocollo diagnostico cambia automaticamente i modelli
usati per generare le tre submission.

## Verifiche e lavoro ancora necessario

```bash
python -m unittest discover -s tests -v
```

Le verifiche sintetiche coprono ordine e allineamento, decisione a 0,5,
coerenza dei conteggi, separazione dei seed, componenti duplicati, raggruppamento
P1 anche nell'encoder, sovrapposizione P2, indipendenza dalle etichette di
validazione, sconosciuti, stato delle trasformazioni, confini della selezione
annidata e confronti sugli stessi gruppi. Sono passati 22 test; la suite che richiede i dati originali è stata saltata.
È stato eseguito anche il confronto globale/interno della CLI con i tre modelli
reali su dati sintetici. Sul medesimo campione, il confronto diretto con il
codice precedente ha confermato matrici e accuracy invariate nel protocollo
originale. È stata verificata la sintassi delle celle del notebook, senza
eseguirlo su Kaggle. Queste verifiche non sono risultati della competizione.

Restano da eseguire sui file originali: audit dei duplicati/ricorrenze, riferimento
singolo, tre ripetizioni, verifica per composizione, confronto globale/interno e
valutazione annidata dei finalisti prescelti. Conservare una cartella distinta
per ogni passaggio e interpretare le differenze soltanto a protocollo, dati e
versione del codice dichiarati.
