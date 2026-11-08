"""Figuras desde historiales y predicciones reales; sin entrenar ni cargar modelos."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / 'PROJECT'
POPULAR = 'cnn_challenge_alt/popular_architectures/'
CUSTOM = 'cnn_challenge_alt/experiments/'
TL = 'tl_challenge/efficientnet_b0/'
(HERE / 'images').mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})
BLUE, ORANGE, GRAY, GREEN = '#264c91', '#d77822', '#9098a4', '#258577'
COLORS = [BLUE, ORANGE, GREEN, '#865ba3', '#a66b38', '#bf4055', '#4b9abd']
provenance = []

def run(path):
    folder = PROJECT / path / 'artifacts'
    m = json.loads((folder / 'metrics.json').read_text())
    c = json.loads((folder / 'config.json').read_text())
    h = json.loads((folder / 'history.json').read_text())
    assert len(h['val_accuracy']) == m['epochs_run']
    assert np.isclose(max(h['val_accuracy']), m['accuracy'], atol=1e-6)
    provenance.append({'run': path, 'accuracy': m['accuracy'], 'f1_macro': m['f1_macro'],
                       'ce': m['valid_inference']['ce'], 'params': m['params'],
                       'best_epoch': m['best_epoch'], 'epochs_run': m['epochs_run'],
                       'image_size': c['image_size']})
    return m, c, h

def num(x):
    return f'{x:.2f}'.replace('.', ',')

def finish(fig, name):
    fig.tight_layout()
    fig.savefig(HERE / 'images' / (name + '.pdf'))
    plt.close(fig)

def learning_curves(runs, name, accuracy_min=None):
    """Puntos originales de history.json, sin suavizado ni interpolación."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3))
    for j, (label, r) in enumerate(runs):
        m, c, h = r
        epochs = np.arange(1, len(h['val_accuracy']) + 1)
        color = COLORS[j]
        axes[0].plot(epochs, np.array(h['val_accuracy'])*100, color=color, lw=1.4, label=label)
        axes[1].plot(epochs, h['val_ce'], color=color, lw=1.4)
        # Mark the selected checkpoint on its own, unextended history.
        axes[0].scatter(m['best_epoch'], m['accuracy']*100, color=color, s=28, zorder=4)
    axes[0].set_ylabel('Accuracy de validación (%)')
    axes[1].set_ylabel('CE de validación')
    if accuracy_min is not None:
        axes[0].set_ylim(bottom=accuracy_min)
    for ax in axes:
        ax.set_xlabel('Época local del entrenamiento')
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=9, loc='lower right')
    finish(fig, name)

popular = {a:run(POPULAR+a+'/experiment01') for a in
           ['resnet18','densenet121','efficientnet_b0','mobilenet_v2']}
res = {1:popular['resnet18'], **{i:run(POPULAR+f'resnet18/experiment{i:02d}') for i in [2,3,4]}}
eff = {1:popular['efficientnet_b0'], **{i:run(POPULAR+f'efficientnet_b0/experiment{i:02d}') for i in [2,3]}}
custom = {i:run(CUSTOM+f'experiment{i:02d}') for i in range(1,6)}
tl = {i:run(TL+f'experiment{i:02d}') for i in range(1,8)}

learning_curves([(label,popular[a]) for label,a in [('ResNet18','resnet18'),
    ('DenseNet121','densenet121'),('EfficientNetB0','efficientnet_b0'),('MobileNetV2','mobilenet_v2')]],
    'populares_curvas')

fig,axes=plt.subplots(1,2,figsize=(12.4,4.3),sharey=True)
for ax,title,runs in [
    (axes[0],'ResNet18',[(1,'Referencia'),(2,'WD 1e−3'),(3,'Smoothing 0.1'),(4,'Smoothing + 192 px')]),
    (axes[1],'EfficientNetB0',[(1,'Referencia'),(2,'LR 3e−4'),(3,'Smoothing 0.1')])]:
    source=res if title=='ResNet18' else eff
    for j,(i,label) in enumerate(runs):
        m,c,h=source[i];x=np.arange(1,len(h['val_accuracy'])+1)
        ax.plot(x,np.array(h['val_accuracy'])*100,color=COLORS[j],lw=1.3,label=label)
        ax.scatter(m['best_epoch'],m['accuracy']*100,color=COLORS[j],s=26)
    ax.set(title=title,xlabel='Época');ax.grid(alpha=.18);ax.legend(fontsize=9,loc='lower right')
axes[0].set_ylabel('Accuracy de validación (%)')
finish(fig,'ajustes_curvas')

learning_curves([(label,custom[i]) for i,label in [(1,'01 · Referencia'),(2,'02 · Smoothing 0.1'),
    (3,'03 · MixUp 0.2'),(4,'04 · MixUp 0.4'),(5,'05 · MixUp 0.2 + 192 px')]],'custom_curvas')

learning_curves([(label,tl[i]) for i,label in [(1,'01 · Solo cabeza entrenable'),(2,'02 · FT parcial'),
    (3,'03 · + MixUp 0.2'),(4,'04 · FT completo salvo BN'),(5,'05 · 192 px'),
    (6,'06 · LR 3e−5'),(7,'07 · MixUp 0.4')]],'tl_todas_curvas')
learning_curves([('Exp06 · MixUp 0.2',tl[6]),('Exp07 · MixUp 0.4',tl[7])], 'tl_curvas')

final=[('B0 desde cero',eff[1]),('ResNet18 · exp04',res[4]),('Custom · exp03',custom[3]),
       ('B0 TL · exp06',tl[6]),('B0 TL · exp07',tl[7])]
fig,axes=plt.subplots(1,2,figsize=(12.4,4.3),sharey=True,gridspec_kw={'width_ratios':[1.4,1]})
colors=[GRAY,GRAY,GREEN,BLUE,ORANGE]
for i,(_,r) in enumerate(final):
    m,c,h=r;v=m['accuracy']*100;params=m['params']/1e6
    axes[0].scatter(v,i,color=colors[i],s=80);axes[0].text(v+.12,i,num(v)+' %',va='center')
    axes[1].barh(i,params,color=colors[i],height=.45);axes[1].text(params+.2,i,num(params),va='center')
axes[0].set(yticks=np.arange(5),yticklabels=[a for a,_ in final],xlim=(77,82.5),xlabel='Accuracy de validación (%)')
axes[1].set(xlim=(0,14),xlabel='Parámetros (millones)')
for ax in axes:ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
axes[0].invert_yaxis();finish(fig,'balance_final')

# Smaller experiment groups for the chronological presentation.
learning_curves([('01 · Referencia',custom[1]),('02 · Smoothing 0.1',custom[2]),
                 ('03 · MixUp 0.2',custom[3])], 'custom_inicio')
learning_curves([('03 · MixUp 0.2, 128 px',custom[3]),('04 · MixUp 0.4, 128 px',custom[4]),
                 ('05 · MixUp 0.2, 192 px',custom[5])], 'custom_final')
learning_curves([('02 · Últimos bloques',tl[2]),('03 · + MixUp 0.2',tl[3]),
                 ('04 · Todos los bloques salvo BN',tl[4])], 'tl_adaptacion')
learning_curves([('05 · 192 px, LR 1e−5',tl[5]),('06 · LR 3e−5',tl[6]),
                 ('07 · MixUp 0.4',tl[7])], 'tl_ajustes_finales')

# Real validation predictions from the selected exp06 checkpoint.
p=PROJECT/TL/'experiment06/artifacts/validation_predictions.npz'
with np.load(p,allow_pickle=False) as data:
    y=data['y_true'];probs=data['probabilities'];saved=data['confusion_matrix']
    cm=np.zeros((13,13),dtype=int)
    np.add.at(cm,(y,probs.argmax(axis=1)),1)
    np.testing.assert_array_equal(cm,saved)
assert np.isclose(np.trace(cm)/cm.sum(),tl[6][0]['accuracy'])
norm=cm/np.maximum(cm.sum(axis=1,keepdims=True),1)*100
classes=tl[6][1]['class_names']
fig,ax=plt.subplots(figsize=(11.8,5.2))
im=ax.imshow(norm,cmap='Blues',vmin=0,vmax=100,aspect='auto')
for i in range(13):
    for j in range(13):
        if i==j or norm[i,j]>=5:
            ax.text(j,i,f'{norm[i,j]:.0f}',ha='center',va='center',fontsize=8,
                    color='white' if norm[i,j]>55 else '#26334a')
ax.set(xticks=np.arange(13),yticks=np.arange(13),xticklabels=classes,yticklabels=classes,
       xlabel='Clase predicha',ylabel='Clase real')
plt.setp(ax.get_xticklabels(),rotation=32,ha='right',fontsize=9)
plt.setp(ax.get_yticklabels(),fontsize=9)
fig.colorbar(im,ax=ax,pad=.02,label='% de cada clase real')
finish(fig,'confusion_tl06')
(HERE/'fuentes_resultados.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
print('Figuras regeneradas desde 21 historiales reales y predicciones guardadas; sin entrenamiento.')
