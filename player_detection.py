"""
    Upravljanje:
       ESC -- izlaz iz programa
       "p" ili "P" -- toggle pauza
       "b" ili "B" -- vraća video 5 frameova unazad
       blank -- ako je pauza, pomiče jedan frame naprijed
"""

from __future__ import print_function
import numpy as np
import cv2
import pandas as pd
import time
import argparse

imeVidea = r"t7.mp4"
imeTrajektorije = r"t7_oznaceno_bez_headera.txt"

najveci_prec=0
najveci_rec=0
precSum=0
recSum=0

def ucitaj_trajektorije_pandas(imeDatoteke):
    """
      učitavanje trajektorija iz datoteke s oznakama
      formira se pandas dataframe s MultiIndexom
        -- da se može dohvaćati slikovni okvir, a unutar njega i pojedini igrači
    """

    df = pd.read_csv(
        imeDatoteke,
        sep=";",
        index_col=[0,1],
        skipinitialspace=True,
        header=None,
        names=['frameID', 'playerID', 'minRow', 'maxRow', 'minCol', 'maxCol', 'unused1', 'unused2', 'unused3', 'unused4'],
        engine='python'
    )

    return df


fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.4
boja = (0, 0, 255)
debljinaOkvira = 1
lista_igraca=[]


def getGTModels(tekuciFrame, frame):
    """
       iz dataframea trajektorije uzima bounding boxove
       iscrtavaju se samo bounding boxovi, ne i repovi trajektorija
    """
    global lista_igraca
    okvir = trajektorije.loc[tekuciFrame]
    for player in okvir.index:
        y1, y2, x1, x2 = okvir.loc[player, slice('minRow', 'maxCol')]
        y1, y2, x1, x2 = int(y1), int(y2), int(x1), int(x2)
        lista_igraca.append((y1,y2,x1,x2))
        #cv2.rectangle(frame, (x1, y1), (x2, y2), boja, debljinaOkvira)
        #tekst = "{}".format(player)
        #cv2.putText(frame, tekst, (x2, y1), fontFace, fontScale, boja)
        
#calculate precision and recall, returns tuple (precision, recall)
def calculate_prec_rec():
    global tp,fp,fn
    global najveci_prec,najveci_rec
    global precSum, recSum
    precision=0
    recall=0
    if(tp+fp)>0:precision=tp/(tp+fp)
    if(tp+fn)>0:recall=tp/(tp+fn)
    precSum+=precision
    recSum+=recall
    najveci_prec=max(najveci_prec,precision)
    najveci_rec=max(najveci_rec,recall)
    return float(precision),float(recall)

#calculate and return sum of last nVal values in buffer and remove them from the buffer
def sumBuffer(buffer,nVal):
    sum=0;i=0
    while i<nVal:
        if not buffer: raise Exception("Prazan buffer, potencijalno neispravne vrijednosti preciznosti i odziva")
        sum+=buffer.pop()
        i+=1
    return sum

def ispisiPodatke(tekuciFrame, frame):
    """
      ispis podataka su status baru na vrhu prozora

    """
    global tp, fp, fn
    global najveci_prec,najveci_rec
    fontScale = 0.5
    hf, wf, channels = frame.shape
    vrijeme = tekuciFrame/25 # sekundi, pretpostavlja se video s 25 FPS
    minute = int(vrijeme/60)
    sekunde = vrijeme - 60*minute # float
    tekst = "frame: {:06} time: {:02}:{:05.2f}  {:.1f} fps".format(tekuciFrame, minute, sekunde, FPS)
    cv2.rectangle(frame, (5, 5), (wf - 5, 25), (0, 0, 0), -1)  # -1 je filled
    cv2.putText(frame, tekst, (20, 20), fontFace, fontScale, (255, 255, 255))
    #eval_tekst = "Precision:{:.4f} Recall:{:.4f} TP:{} FP:{} FN:{} MP:{:.4f} MR:{:.4f}".format(precision,recall, tp, fp, fn,najveci_prec,najveci_rec)
    #cv2.rectangle(frame, (5, hf-25), (round(wf/2)-500, hf), (0, 0, 0), -1)  # -1 je filled
    #cv2.putText(frame, eval_tekst, (20, hf-10), fontFace, fontScale, (255, 255, 255))
    
def obradiSliku(frameDraw, scale_percent):
    y=frameDraw.shape[0]
    x=frameDraw.shape[1]
    frameDraw = frameDraw[50:y-100,500:x-100]
    width = int(x * scale_percent / 100)
    height = int(y * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frameDraw, dim, interpolation = cv2.INTER_AREA)
    return resized

def redraw():
    global lista_igraca
    lista_igraca=[]
    frameDraw = frame.copy()
    getGTModels(tekuciFrame,frameDraw)
    drawBBox(frameDraw)
    resized=obradiSliku(frameDraw, 70)
    ispisiPodatke(tekuciFrame, resized)
    cv2.imshow("frame", resized)

#function containing actions to do in each new frame
def forwardFrame():
    global fgMask
    global frame, tekuciFrame, trajektorije
    global precisBuf,recallBuf, maxSize
    ret, frame = cap.read()
    if not ret: return ret
    tekuciFrame += 1
    fgMask = backSub.apply(frame)
    fgMask[fgMask==127]=0 #micemo sjene
    precision,recall=calculate_prec_rec()
    if len(precisBuf)==maxSize:
        precisBuf.pop(0)
    precisBuf.append(precision)
    if len(recallBuf)==maxSize:
        recallBuf.pop(0)
    recallBuf.append(recall)
    
    global vrijeme1, vrijeme2, FPS
    if tekuciFrame%100 == 0:
        vrijeme1 = vrijeme2
        vrijeme2 = time.time()
        if vrijeme2 > vrijeme1:
          FPS = 100/(vrijeme2 - vrijeme1)

    return ret

#calculate Jaccard index (IoU) of bounding boxes A and B
def bb_iou(boxA, boxB):
	
	yA = max(boxA[0], boxB[0])
	yB = min(boxA[1], boxB[1])
	xA = max(boxA[2], boxB[2])
	xB = min(boxA[3], boxB[3])

	interArea = max(0, xB - xA+1) * max(0, yB - yA+1) 
	boxAArea = (boxA[3] - boxA[2]+1) * (boxA[1] - boxA[0]+1)
	boxBArea = (boxB[3] - boxB[2]+1) * (boxB[1] - boxB[0]+1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	
	return iou

def evaluateDetection(iou_thresh,windowTuple):
    #tuple structure for player:(y1,y2,x1,x2)
    global tp,fp,fn
    pronasaoIgraca=False
    x,y,w,h=windowTuple
    box_detect=(y,y+h,x,x+w)
    for igrac in lista_igraca:
        iou=bb_iou(box_detect,igrac)
        if iou!=0: 
            pronasaoIgraca=True
            if iou<iou_thresh:
                #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)#debug
                #tekst = "IOU:{:.4f}".format(iou)
                #cv2.putText(frame, tekst, (x, y), fontFace, fontScale, (255,0,255))
                fp+=1
            else:
                #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) #debug
                #tekst = "IOU:{:.4f}".format(iou)
                #cv2.putText(frame, tekst, (x, y), fontFace, fontScale, boja)
                tp+=1
            break
    if pronasaoIgraca==False:fp+=1
    fn=len(lista_igraca)-tp   
        
def resetStats():
    global tp, fp, fn
    tp=0;fp=0;fn=0

#predict all bounding boxes around players and draw them
def drawBBox(frame):
    
    global kernel, kernel_dil, kernel_erod,fgMask
    
    resetStats()
    fgMask=cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    dilation = cv2.erode(fgMask, kernel_erod, iterations=1)
    dilation = cv2.dilate(dilation, kernel_dil, iterations=1)
    #cv2.imshow("dilation", obradiSliku(dilation, 70))
    contours,hierarchy=cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area=cv2.contourArea(contour)
        if(area>300 and area<1250):
            x,y,w,h=cv2.boundingRect(contour)
            if(w<h):
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                evaluateDetection(0.5,(x,y,w,h))
        
    
# ------------------------------------
#  POČETAK

# Parsiraj argumente iz komandne linije
parser = argparse.ArgumentParser(description='This program is used for football player detection using background subtraction.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='t7.mp4')
parser.add_argument('--gt', type=str, help='Path to a player ground_truth frame file.', default='t7_oznaceno_bez_headera.txt')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
args = parser.parse_args()
#-------------------------------------
#
#alg=input("Unesite željeni algoritam (KNN ili MOG2): ")
#if alg.lower() == 'knn':
#    backSub = cv2.createBackgroundSubtractorKNN()
#else:
#    backSub = cv2.createBackgroundSubtractorMOG2(128,cv2.THRESH_BINARY,1)
#
if args.algo == 'MOG2':
    backSub = cv2.createBackgroundSubtractorMOG2(128,cv2.THRESH_BINARY,1)
else:
    backSub = cv2.createBackgroundSubtractorKNN()
imeTrajektorije=args.gt
starttime = time.time()
trajektorije = ucitaj_trajektorije_pandas(imeTrajektorije)
totaltime = time.time()-starttime
print("učitavanje trajektorija: {} sec ".format(totaltime))

## [capture]
cap = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))
if not cap.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)
## [capture]
Pauza = True
tekuciFrame = 0

#pohrani vrijednosti preciznosti i odziva za prethodnih maxSize okvira
recallBuf=list()
precisBuf=list()
maxSize=120

FPS = 10  # početna nerelevantna vrijednost
vrijeme2 = time.time()
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel_dil=np.ones((4,4),np.uint8)
kernel_erod=np.ones((2,2),np.uint8)
tp=0;fp=0;fn=0
while(1):
    if not Pauza or tekuciFrame == 0:
        ret = forwardFrame()
        if not ret:
            break
    
    redraw()

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        # prekid programa s ESC (ASCII 27)
        break
    elif k == ord("p") or k == ord("P"):
        Pauza = not Pauza
    elif k == ord("b") or k == ord("B"):
        # premotamo video nazad za 5 frameova (ako ide)
        if tekuciFrame > 6:
            tekuciFrame -= 6
            precSum-=sumBuffer(precisBuf,6)
            recSum-=sumBuffer(recallBuf,6)
            cap.set(cv2.CAP_PROP_POS_FRAMES, tekuciFrame)
            ret = forwardFrame()
            if not ret:
                break
    elif k == ord(" ") and Pauza:
        ret = forwardFrame()
        if not ret:
            break

cv2.destroyAllWindows()
cap.release()
print()
print("Average precision:",precSum/tekuciFrame)
print("Average recall:",recSum/tekuciFrame)
print()
print("Max precision:",najveci_prec)
print("Max recall:",najveci_rec)