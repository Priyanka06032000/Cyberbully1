import pandas as pd, numpy as np, re
import speech_recognition as sr
import pyttsx3
from sklearn.metrics import classification_report, accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
import tkinter as tk
from sklearn import svm
from PIL import Image, ImageTk
from tkinter import ttk
from joblib import dump , load
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pickle
import nltk
#######################################################################################################
nltk.download('stopwords')
stop = stopwords.words('english')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#######################################################################################################
    
root = tk.Tk()
root.title("cyber bullying using twitter")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
img=ImageTk.PhotoImage(Image.open("img.jpg"))

img2=ImageTk.PhotoImage(Image.open("y91.jpg"))

img3=ImageTk.PhotoImage(Image.open("bg7.jpg"))


logo_label=tk.Label()
logo_label.place(x=0,y=0)

x = 1

# function to change to next image
def move():
	global x
	if x == 4:
		x = 1
	if x == 1:
		logo_label.config(image=img)
	elif x == 2:
		logo_label.config(image=img2)
	elif x == 3:
		logo_label.config(image=img3)
	x = x+1
	root.after(2000, move)

# calling the function
move()
#background_label.place(x=0, y=0)

###########################################################################################################
lbl = tk.Label(root, text="Cyber Bullying Using Twitter", font=('times', 35,' bold '), height=2, width=50,bg="black",fg="white")
lbl.place(x=0, y=0)
##############################################################################################################################


def Data_Display():
    columns = ['Article_link', 'Headline', 'is_spam']
    print(columns)

    data1 = pd.read_csv("D:/cyber bullying1/cyber bullying/dataset4.csv", encoding='unicode_escape')

    data1.shape

    data1.shape

    data1.head()

    data1

    data1

    article_link = data1.iloc[:, 0]
    headline = data1.iloc[:, 1]
    is_spam = data1.iloc[:, 2]


    display = tk.LabelFrame(root, width=100, height=400, )
    display.place(x=270, y=100)

    tree = ttk.Treeview(display, columns=(
    'Article_link', 'Headline', 'is_spam'))

    style = ttk.Style()
    style.configure('Treeview', rowheight=40)
    style.configure("Treeview.Heading", font=("Tempus Sans ITC", 15, "bold italic"))
    style.configure(".", font=('Calibri', 10), background="black")
    style.configure("Treeview", foreground='white', background="black")

    tree["columns"] = ("1", "2", "3")
    tree.column("1", width=130)
    tree.column("2", width=150)
    tree.column("3", width=200)

    tree.heading("1", text="Article_link")
    tree.heading("2", text="Headline")
    tree.heading("3", text="is_spam")

    treeview = tree

    tree.grid(row=0, column=0, sticky=tk.NSEW)

    print("Data Displayed")

    for i in range(0, 304):
        tree.insert("", 'end', values=(
        article_link[i], headline[i], is_spam[i]))
        i = i + 1
        print(i)

##############################################################################################################


def Train():
    
    result = pd.read_csv(r"D:/cyber bullying1/cyber bullying/ct1.csv",encoding = 'unicode_escape')

    result.head()
        
    result['headline_without_stopwords'] = result['headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    
    def pos(review_without_stopwords):
        return TextBlob(review_without_stopwords).tags
    
    
    os = result.headline_without_stopwords.apply(pos)
    os1 = pd.DataFrame(os)
    #
    os1.head()
    
    os1['pos'] = os1['headline_without_stopwords'].map(lambda x: " ".join(["/".join(x) for x in x]))
    
    result = result = pd.merge(result, os1, right_index=True, left_index=True)
    result.head()
    result['pos']
    review_train, review_test, label_train, label_test = train_test_split(result['pos'], result['is_spam'],
                                                                              test_size=0.30, random_state=1234)
    
    tf_vect = TfidfVectorizer(lowercase=True, use_idf=True, smooth_idf=True, sublinear_tf=False)
    
    X_train_tf = tf_vect.fit_transform(review_train)
    X_test_tf = tf_vect.transform(review_test)
    
    
    def svc_param_selection(X, y, nfolds):
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma': gammas}
        grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)
        grid_search.fit(X, y)
        return grid_search.best_params_
    
    
    svc_param_selection(X_train_tf, label_train, 5)
    #
    
    clf = svm.SVC(C=10, gamma=0.001, kernel='linear')   
    clf.fit(X_train_tf, label_train)
    pred = clf.predict(X_test_tf)
    
    with open('vectorizer.pickle', 'wb') as fin:
        pickle.dump(tf_vect, fin)
    with open('mlmodel.pickle', 'wb') as f:
        pickle.dump(clf, f)
    
    pkl = open('mlmodel.pickle', 'rb')
    clf = pickle.load(pkl)
    vec = open('vectorizer.pickle', 'rb')
    tf_vect = pickle.load(vec)
    
    X_test_tf = tf_vect.transform(review_test)
    pred = clf.predict(X_test_tf)
    
    print(metrics.accuracy_score(label_test, pred))
    
    print(confusion_matrix(label_test, pred))
    
    print(classification_report(label_test, pred))

       
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(label_test, pred)))
    print("Accuracy : ",accuracy_score(label_test, pred)*100)
    accuracy = accuracy_score(label_test, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(label_test, pred) * 100)
    repo = (classification_report(label_test, pred))
    
    label4 = tk.Label(root,text =str(repo),width=35,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=205,y=100)
    
    label5 = tk.Label(root,text ="Accracy : "+str(ACC)+"%\nModel saved as SVM_MODEL.joblib",width=35,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=205,y=320)
    
    dump (clf,"SVM_MODEL.joblib")
    print("Model saved as SVM_MODEL.joblib")

################################################################################################################################################################



entry = tk.Entry(root,width=50,font=("Times new roman",20,"bold"))
entry.insert(0,"Enter text here...")
entry.place(x=400,y=200)
##############################################################################################################################################################################
def Test():
    predictor = load("SVM_MODEL.joblib")
    Given_text = entry.get()
    #Given_text = "the 'roseanne' revival catches up to our thorny po..."
    vec = open('vectorizer.pickle', 'rb')
    tf_vect = pickle.load(vec)
    X_test_tf = tf_vect.transform([Given_text])
    y_predict = predictor.predict(X_test_tf)
    print(y_predict[0])
    if y_predict[0]==0:
        label4 = tk.Label(root,text ="Normal Sentence",width=20,height=2,bg='#46C646',fg='black',font=("Tempus Sanc ITC",25))
        label4.place(x=450,y=550)
    elif y_predict[0]==1:
        label4 = tk.Label(root,text ="Gender bullying  Sentence",width=20,height=2,bg='#FF3C3C',fg='black',font=("Tempus Sanc ITC",25))
        label4.place(x=450,y=550)
    elif y_predict[0]==2:
        label4 = tk.Label(root,text ="Religion bullying  Sentence",width=20,height=2,bg='#FF3C3C',fg='black',font=("Tempus Sanc ITC",25))
        label4.place(x=450,y=550)
    elif y_predict[0]==3:
        label4 = tk.Label(root,text =" Other bullying  Sentence",width=20,height=2,bg='#FF3C3C',fg='black',font=("Tempus Sanc ITC",25))
        label4.place(x=450,y=550)
    elif y_predict[0]==4:
        label4 = tk.Label(root,text =" Age bullying  Sentence",width=20,height=2,bg='#FF3C3C',fg='black',font=("Tempus Sanc ITC",25))
        label4.place(x=450,y=550)
    elif y_predict[0]==5:
        label4 = tk.Label(root,text =" Ethnicity bullying  Sentence",width=20,height=2,bg='#FF3C3C',fg='black',font=("Tempus Sanc ITC",25))
        label4.place(x=450,y=550)

def speech_text():
    r = sr.Recognizer()
    def SpeakText(command):
        engine = pyttsx3.init()
        engine.say(command)
        engine.runAndWait() 
    while(1):   
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
                print("Did you say "+MyText)
                SpeakText(MyText)
                return MyText
    
def Testaudio():
    r = sr.Recognizer()
     
    # Function to convert text to
    # speech
    def SpeakText(command):
         
        # Initialize the engine
        engine = pyttsx3.init()
        engine.say(command)
        engine.runAndWait()
        
         
         
    # Loop infinitely for user to
    # speak
     
    while(1):   
         
        # Exception handling to handle
        # exceptions at the runtime
        try:
             
            # use the microphone as source for input.
            with sr.Microphone() as source2:
                 
                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level
                r.adjust_for_ambient_noise(source2, duration=0.2)
                 
                #listens for the user's input
                audio2 = r.listen(source2)
                 
                # Using google to recognize audio
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
     
                print("Did you say "+MyText)
                SpeakText(MyText)
                abc=MyText
                predictor = load("SVM_MODEL.joblib")
                Given_text = abc
                            #Given_text = "the 'roseanne' revival catches up to our thorny po..."
                vec = open('vectorizer.pickle', 'rb')
                tf_vect = pickle.load(vec)
                X_test_tf = tf_vect.transform([Given_text])
                y_predict = predictor.predict(X_test_tf)
                print(y_predict[0])
                if y_predict[0]==0:
                    label4 = tk.Label(root,text ="Normal Sentence",width=20,height=2,bg='#46C646',fg='black',font=("Tempus Sanc ITC",25))
                    label4.place(x=450,y=550)
                elif y_predict[0]==1:
                    label4 = tk.Label(root,text ="Gender bullying  Sentence",width=20,height=2,bg='#FF3C3C',fg='black',font=("Tempus Sanc ITC",25))
                    label4.place(x=450,y=550)
                elif y_predict[0]==2:
                    label4 = tk.Label(root,text ="Religion bullying  Sentence",width=20,height=2,bg='#FF3C3C',fg='black',font=("Tempus Sanc ITC",25))
                    label4.place(x=450,y=550)
                elif y_predict[0]==3:
                    label4 = tk.Label(root,text =" Other bullying  Sentence",width=20,height=2,bg='#FF3C3C',fg='black',font=("Tempus Sanc ITC",25))
                    label4.place(x=450,y=550)
                elif y_predict[0]==4:
                    label4 = tk.Label(root,text =" Age bullying  Sentence",width=20,height=2,bg='#FF3C3C',fg='black',font=("Tempus Sanc ITC",25))
                    label4.place(x=450,y=550)
                elif y_predict[0]==5:
                    label4 = tk.Label(root,text =" Ethnicity bullying  Sentence",width=20,height=2,bg='#FF3C3C',fg='black',font=("Tempus Sanc ITC",25))
                    label4.place(x=450,y=550)    

                 
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
             
        except sr.UnknownValueError:
            print("unknown error occured")
    
    

def window():
    root.destroy()
    
# button1 = tk.Button(frame,command=Data_Display,text="Data_Display",bg="#E46EE4",fg="white",width=15,font=("Times New Roman",15,"bold"))
# button1.place(x=25,y=30)

# button2 = tk.Button(frame,command=Train,text="Train",bg="#E46EE4",fg="white",width=15,font=("Times New Roman",15,"bold"))
# button2.place(x=25,y=100)

button3 = tk.Button(root,command=Test,text="Test by Text",bg="#E46EE4",fg="white",width=15,font=("Times New Roman",15,"bold"))
button3.place(x=600,y=250)

#button3 = tk.Button(frame,command=Testaudio,text="Test by Audio",bg="#E46EE4",fg="white",width=15,font=("Times New Roman",15,"bold"))
#button3.place(x=25,y=290)

button4 = tk.Button(root,command=window,text="Exit",bg="#E46EE4",fg="white",width=15,font=("Times New Roman",15,"bold"))
button4.place(x=600,y=300)




root.mainloop()