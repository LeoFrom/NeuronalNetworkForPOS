#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from math import exp
from random import shuffle


epsilon = 1e-5


class Tanh:
	"""
	Fonction d'activation Tanh
	"""

    def __init__(self):
        self.nom="tanh"
        
    def calcul(self, x):
        return np.tanh(x)

    def derivee(self, x):
          
        def aux_dev(y):
            return 1-np.power(np.tanh(y),2)
              
        return np.apply_along_axis(aux_dev, 0, x).reshape(x.shape)


class Relu:
	"""
	Fonction d'activation Relu
	"""
  
    def __init__(self):
        self.nom="relu"

    def calcul(self, x):
          
        def aux(y):
            return max(0,y)
              
        return np.apply_along_axis(aux,0,x).reshape(x.shape)

    def derivee(self, x):
          
        return np.array([np.float(x[0,i]>0) for i in range(x.shape[1])]).reshape(x.shape)


class Softmax:
	"""
	Fonction d'activation Softmax
	"""

    def __init__(self):
        self.nom="softmax"

    def calcul(self, x):
        deno = np.sum(np.exp(x))
              
        return np.exp(x)/deno

    def derivee(self, x, wrt_i):
          
        if wrt_i >= x.shape[1] or wrt_i < 0:
            print ("error", x, wrt_i)
            return
                
        else:
            temp = self.calcul(x)
            res =[]
                        
            for i in range(x.shape[1]):
                if i==wrt_i:
                    res.append(temp[0, wrt_i]*(1-temp[0, wrt_i]))
                else :
                    res.append(-temp[0, i]*temp[0, wrt_i])
                        
        return temp[0, wrt_i]*(1-temp[0, wrt_i])







class Layer:
        """ Classe permettant de créer un couche dans un réseau de neuronne
		
		Parametres:
		taille_entree: la taille de la couche en entrée 
		taille_sortie: la taille de la couche en sortie
		activation: la fonction d'activation
		
        """
        
        
        def __init__(self, taille_entree, taille_sortie, activation=None):
                
                self.outputs = np.zeros((1,taille_sortie))                              #
                self.consumers = None                                                   #
                self.precedents = None                                                  #
                self.activation = activation                                            #
                
                #
                if self.activation == None:
                        a= np.sqrt(6/(taille_entree+taille_sortie))
                        self.W = np.random.uniform(-a,a,(taille_entree, taille_sortie)) #
                        self.B = np.zeros((1,taille_sortie))                            #
                        

        def forward(self, x,shape=None):
          
                """ Fonction forward de la couche qui renvoie la couche de neurones.
  
                Parametres:
                x: np.array; vecteur en entré de la couche.
          
                Returns:
                x: np.array; le vecteur de sortie de la couche.  """
          
                #
                if self.activation == None :

                        #pour le checking
                        if shape!=None:
                                add=np.zeros(self.W.shape)
                                add[shape[0],shape[1]]=epsilon
                                self.outputs = np.matmul(x,self.W+add + self.B) # combinaison lineaire
                        else:
                                self.outputs = np.matmul(x,self.W) + self.B     # combinaison lineaire
                #
                else:
                        self.outputs = self.activation.calcul(x)                # fonction d'activation

                if np.array_equal(self.outputs,np.ones(self.outputs.shape)):
                        print(self.outputs,"????????????????????????????????????????????????????????????????????????")
                
                return self.outputs

                                
        def backprop_mat(self, gradient, gold=None):
                """ Calcul la matrice obtenu par backprop selon le gradient ou la derivee
				
				Parametres:
				gradient: gradient utilisé lors de la backprop
				
				Returns:
				Soit le vecteur entre le gradient de la couche suivante et la derivée de la couche suivante appliqué au vecteur de la couche actuelle
				Soit la matrice entre le gradient et une matrice de poids
                
                """
                                                        
                
                # Si la couche est de pre activation, le gradient est le produit terme à terme entre le gradient
                # de la couche suivante et la derivee de la couche suivante appliqué au vecteur de la couche actuelle
                if self.activation == None:
                        
                        if gold!=None:  
                                return np.multiply(gradient,self.consumers.activation.derivee(self.outputs, gold).T)
                        else:
                                return np.multiply(gradient,self.consumers.activation.derivee(self.outputs).T)
                
                # Sinon produit matriciel entre le gradient de la couche suivante
                # et la matrice de poids de la couche suivante.
                else:
                        return np.matmul(self.consumers.W, gradient)

        def addConsumers(self, layer):
                self.consumers = layer
                layer.precedents = self



class Embeddings_layer:

        """ Classe permettant de créer un couche d'embeddings dans un réseau de neuronne
		
		Parametres:
		n1: le nombre de vecteurs de words 
		nb_embeddings1: nombre total de words
		nb_feats1: le nombre de features des words
		
		n2: le nombre de vecteurs de tags 
		nb_embeddings2: nombre total de tags
		nb_feats2: le nombre de features des tags

        """

        def __init__(self, n1, nb_embeddings1, nb_feats1, n2=0, nb_embeddings2=0, nb_feats2=0):
                
                self.outputs =np.zeros((1,n1*nb_feats1+n2*nb_feats2))           #
                self.consumers = None                                           #
                self.n1=n1                                                      # n1 represente le nombre de words embeddings
                self.nb_feats1=nb_feats1                                        #
                self.embeddings1=np.zeros((nb_embeddings1,nb_feats1))           #
                
                self.n2=n2                                                      #n 2 represente le nombre de tag embeddings
                if self.n2:
                        self.nb_feats2=nb_feats2                                #
                        self.embeddings2=np.zeros((nb_embeddings2,nb_feats2))   #
                 
        def forward(self, x):
          
                """ Fonction forward de la couche qui renvoie la concatenation des embeddings.
  
                Parametres:
                x: np.array; vecteur en entré du reseau, qui est simplement la suite des id .
          
                Returns:
                x: np.array; le vecteur concatene des embeddings.  
				"""
                
                v=[]                                                            #
                
                #
                for i in range(self.n1):
                        v.append(self.embeddings1[x[i],:].copy())
                    
                if self.n2:
                        #
                        for j in range(self.n2):
                                v.append(self.embeddings2[x[self.n1+j],:].copy())
                
                #concatenation des embeddings                
                self.outputs=np.concatenate(v).reshape(self.outputs.shape)
                
                return self.outputs
        
        def backprop_embeddings(self, gradient):
                """  Retourne la matrice entre le gradient et le vecteur d'observation
				
				Parametres:
				gradient : le gradient de la backprop
                
                """
                return np.matmul(self.consumers.W, gradient)

        def addConsumers(self, layer):
                self.consumers = layer
                layer.precedents = self
                                        

class Network:

        """ Classe qui construit un réseau de neuronnes

		Parametres:
		taille_entree: la taille en entrée de la couche d'entrée
		taille_sortie: la taille en sortie de la couche de sortie
		activation_hidden: la fonction d'activation sur les couches cachées
		activation_sortie: la fonction d'activation sur la couche de sortie
		nb_hiddens: le nombre de couches cachées
		layers_size: la taille des couches en dehors de celle d'entrée et celle de sortie
		n1: le nombre de vecteurs de words 
		nb_embeddings1: nombre total de words
		nb_feats1: le nombre de features des words
		n2: le nombre de vecteurs de tags 
		nb_embeddings2: nombre total de tags
		nb_feats2: le nombre de features des tags

        """

        def __init__(self, taille_entree, taille_sortie, activation_hidden, activation_sortie, nb_hiddens, layers_size,
                     n1=0, nb_embeddings1=0, nb_feats1=0, n2=0, nb_embeddings2=0, nb_feats2=0 ):


                #taille des inputs du réseau
                self.taille_entree = taille_entree                              # taille des inputs du réseau
                self.taille_sortie = taille_sortie                              # taille du vecteur en sortie du réseau, cad nombre de classes du probleme
                self.activation_hidden = activation_hidden                      # fonction d'activation des couches cachées
                self.activation_sortie = activation_sortie                      # fonction d'activation en sortie
                self.nb_hiddens=nb_hiddens                                      # nombre de paire de couches (pre activation et activation) cachées dans le réseau
                self.layers_size=layers_size                                    # taille des couches cachées              
                self.layers = []                                                  #liste des couches du reseau
                self.n1 = n1
                self.nb_embeddings1 = nb_embeddings1
                self.nb_feats1 = nb_feats1
                self.n2 = n2
                self.nb_embeddings2 = nb_embeddings2
                self.nb_feats2 = nb_feats2 

                
                # Si nécessite une couche d'embeddings
                if self.n1:
                        # ajout d'une couche d'embeddings
                        self.layers.append(Embeddings_layer(n1, nb_embeddings1, nb_feats1, n2, nb_embeddings2, nb_feats2 ))       

                        # ajout de la premiere couche cachee de pre activation
                        self.layers.append(Layer(n1*nb_feats1+n2*nb_feats2, self.layers_size))
                        self.layers[-2].addConsumers(self.layers[-1])
                        
                        # ajout de la premiere couche cachee d' activation
                        self.layers.append(Layer(self.layers_size,self.layers_size, self.activation_hidden))
                        self.layers[-2].addConsumers(self.layers[-1])
                
                # Sinon ajout direct de la premiere couche cachee
                else:
                        # ajout de la premiere couche cachee de pre activation
                        self.layers.append(Layer(self.taille_entree, self.layers_size))
                      
                        # ajout de la premiere couche cachee d' activation
                        self.layers.append(Layer(self.layers_size,self.layers_size, self.activation_hidden))
                        self.layers[-2].addConsumers(self.layers[-1])

                # ajout des couches cachees restantes
                for _ in range(1,nb_hiddens):
                  
                        # ajout de la k_ieme couche cachee de pre activation
                        self.layers.append(Layer(self.layers_size, self.layers_size))
                        self.layers[-2].addConsumers(self.layers[-1])
                        
                        # ajout de la k_ieme couche cachee d'activation
                        self.layers.append(Layer(self.layers_size,self.layers_size, self.activation_hidden))
                        self.layers[-2].addConsumers(self.layers[-1])
                
                # ajout de la couche cachee de pre activation de sortie
                self.layers.append(Layer(self.layers_size,self.taille_sortie))
                self.layers[-2].addConsumers(self.layers[-1])
                
                # ajout de la couche cachee d'activation de sortie
                self.layers.append(Layer(self.taille_sortie,self.taille_sortie, self.activation_sortie))
                self.layers[-2].addConsumers(self.layers[-1])


        def forward(self,x, where=None):
                """ Fonction forward du reseau qui calcule iterativement la sortie du reseau
                en appelant les fonctions forwards des couches.
  
                Parametres:
                x: np.array; vecteur en entré du reseau.
                
                Si checking du gradient
                where: tuple; indice rangé, indice colonne pour rajout de la variable epsilon du checking
          
                Returns:
                x: np.array; le vecteur en sortie du réseau.  """
                
                #boucle pour appeller la méthode forward de toute les couches
                for i in range(len(self.layers)):
                  
                        if where!=None and i==where[0]:
                                x=self.layers[i].forward(x,(where[1],where[2]))
                        else:
                                x=self.layers[i].forward(x)
                        
                
                return x
              
              
        def save(self):
                """ Sauvegarde le model pour lequel on a obtenu des bonnes valeurs
				
				Returns:
				model : retourne le model (de réseau de neurone) correspondant à celui pour lequel on a eu un bon résultat
                
                """
          
                model = {}
                hyper_parameters = {"taille_entree": self.taille_entree, "taille_sortie" : self.taille_sortie, "activation_hidden" : self.activation_hidden,
                              "activation_sortie" : self.activation_sortie, "nb_hiddens" : self.nb_hiddens, "layers_size" : self.layers_size ,
                              "n1" :self.n1 , "nb_embeddings1" : self.nb_embeddings1, "nb_feats1": self.nb_feats1, "n2" : self.n2 , 
                              "nb_embeddings2" : self.nb_embeddings2, "nb_feats2" : self.nb_feats2   }
                
                model["hyper_parameters"] = hyper_parameters
                
                theta= []
                deb = 0
                
                if self.n1:
                    deb = 1
                    model["embeddings"] = []
                    model["embeddings"].append(np.copy(self.layers[0].embeddings1))
                
                if self.n2:
                    model["embeddings"].append(np.copy(self.layers[0].embeddings2))
              
              
                for i in range( deb, len(self.layers), 2):
                    W, B = np.copy(self.layers[i].W),  np.copy(self.layers[i].B)
                    theta.append([W,B])
          
                model["params"]=theta

                return model
        
        
        def setParams(self, params):
                """ Set les paramètres (couche du réseau) correspondant à des paramètres donnant un bon résultat
                """
          
                if (len(params)*2 != len(self.layers) and not(self.n1)) or (len(params)*2 + 1 != len(self.layers) and self.n1):
                        print("erreur taille")
                        return
          
                deb = 0
                if self.n1:
                        deb = 1
            
                        for i in range(len(params)):
                                if self.layers[deb + i*2].W.shape == params[i][0].shape:
                                        self.layers[deb + i*2].W = np.copy(params[i][0])
                                else:
                                        print("erreur dimension W",i)
                                        return
            
                                if self.layers[deb + i*2].B.shape == params[i][1].shape:
                                        self.layers[deb + i*2].B = np.copy(params[i][1])
                                else:
                                        print("erreur dimension B",i)
                                        return
            
        
        def setEmbeddings(self, embeddings):
                """ Set les embeddings (couche d'embeddings) correspondant à des embeddings donnant un bon résultat
                """
          
                if not(self.n1) or len(embeddings)>2:
                        print("pas compatible n1")
                        return
          
                if self.n2 and len(embeddings)!=2:
                        print("pas compatible n2")
                        return
          
          
                if self.layers[0].embeddings1.shape == embeddings[0].shape :
                        self.layers[0].embeddings1 = np.copy(embeddings[0])
            
                        if self.n2 and self.layers[0].embeddings2.shape == embeddings[1].shape:
                                self.layers[0].embeddings2 = np.copy(embeddings[1])
              
                        elif self.n2: 
                                print("erreur dimension embeddings2")
                                return
                        
                else:
                        print("erreur dimension embeddings1")
                        return

############   Fin classe Network    ####################

def fromModel(dict_model):

    """ Permet de récupérer un modèle (de réseau de neurone) qui nous a permis d'obtenir un bon résultat
	
	dict_model:
	une dictionnaire contenant un modèle (de réseau de neurone) que l'on a sauvegardé au préalable
	
	Returns:
	model: le model(de réseau de neurone) que l'on cherche à récuperer
        
        
    """
  
    model = Network (**dict_model["hyper_parameters"])
  
    model.setEmbeddings( dict_model["embeddings"])
    model.setParams(dict_model["params"])
  
    return model
          
         
            

def NegativeLogLikelihood(v, gold):
  
        """ Fonction de perte correspondant à la log likelihood
        
		Paramètres:
		gold: classe gold
		v: un vecteur
		
		Returns:
		La log likelihood calculée avec la classe gold et le vecteur
        
        """
        return -np.log(v[0,gold])

                


class SGD:
        """ Fonction de descente de gradient stochastique
		
		Parametres:
		model : le modèle (de réseau de neurone) sur lequel on applique la SDG
		lr: le learning rate
		embeddings_layers : si oui ou non on prends en compte les embeddings
		tag_layers: si oui ou non on prends en compte les tags
		suivi : si oui ou non on veut un suivi
		checking : ??????
		threshold : ????
        """
        
        
        def __init__(self, model, lr, embeddings_layers=False, tag_layers=False,  suivi=False, checking=False, threshold=0.1):
                
                self.model=model                                                #
                self.lr=lr                                                      #
                self.embeddings_layers=embeddings_layers                        #
                self.tag_layers=tag_layers                                      #
                
                self.suivi = suivi                                              
                self.checking=checking                                          
                self.threshold=threshold                                        
                
                        
        def checking_gradient(self, gold,x,out):
          
                """ a commenter by ben
                
                """
                gradient=[]                                                     
              
                if self.embeddings_layers:
                        k=1
                else:
                        k=0
                for n in range(self.model.layers[k].W.shape[0]):
                        row=[]
                        for m in range(self.model.layers[k].W.shape[1]):
                                row.append((NegativeLogLikelihood(self.model.forward(x,(k,n,m)),gold)-NegativeLogLikelihood(out,gold))/epsilon)
                        gradient.append(np.array(row))
                gradient=np.array(gradient)
                #print(gradient.shape,self.model.layers[k].W.shape)
                return gradient
                                
                                
        def step(self,gradient, gold, x, checking=None):
          

                
                gradientB, gradientW = 0, 0
                gradientpred=np.copy(gradient)
                
                gradient = self.model.layers[len(self.model.layers)-2].backprop_mat(gradient,gold)                  #gradient de la couche de pré activation de sortie
                if type(self.model.activation_hidden)==type(Relu()) and  np.linalg.norm(gradient) > self.threshold:
                        gradient = gradient * self.threshold/np.linalg.norm(gradient)
                #print(gold, gradient, gradient.shape, len(self.model.layers)-2,"\n")
                if self.suivi:
                        print("sgd")
                        print(gradient)
                

                # boucle de calcul du gradient, de la derniere couche d'activation avant les deux couches de sorties,
                #jusqu'à la toute première couche du réseau.
                for k in range(len(self.model.layers)-3,-1,-1):
                        
                                
                        if self.embeddings_layers and k == 0:
                                gradientpred=np.copy(gradient)
                                gradient=self.model.layers[k].backprop_embeddings(gradient)

                                
                                e=0
                                #boucle pour words embeddings
                                for i in range(self.model.layers[k].n1):
                                        self.model.layers[k].embeddings1[x[i],:] = self.model.layers[k].embeddings1[x[i],:]
                                        - self.lr*gradient[e:e+self.model.layers[k].nb_feats1].reshape(self.model.layers[k].embeddings1[x[i],:].shape)
                                        e += self.model.layers[k].nb_feats1


                                #boucle pour tag embeddings
                                if self.model.layers[k].n2:
                                        for j in range(n2):
                                                self.model.layers[k].embeddings2[x[self.model.layers[k].n1+j],:] = self.embeddings2[x[self.model.layers[k].n1+j],:]
                                                - self.lr*gradient[e:e+self.model.layers[k].nb_feats2]

                                                e += self.model.layers[k].nb_feats2
                        else:
                                #calcul du gradient selon la méthode backprop de la couche
                                gradientpred=np.copy(gradient)
                                gradient = self.model.layers[k].backprop_mat(gradient)
                                if type(self.model.activation_hidden)==type(Relu()) and  np.linalg.norm(gradient) > self.threshold:
                                        gradient = gradient * self.threshold/np.linalg.norm(gradient)
                                if self.suivi:
                                        print(gradient)

                                #mise à jour des poids et biais ( dans le if et else)
                                if self.model.layers[k].activation == None:

                                        # mise à jour des biais
                                        self.model.layers[k].B = self.model.layers[k].B - self.lr * gradient.T
                                                                             
                                        if k > 0:
                                                #calcul du gradient de W, mais comme W est necessaire dans le calcul du gradient de la couche précedente,
                                                #on ne met pas tout de suite à jour W.                                                      
                                                gradientW= np.matmul(gradient,self.model.layers[k].precedents.outputs).T                                                    
                                                    
                                                #self.model.layers[k].W = self.model.layers[k].W - self.lr * np.matmul(gradient,self.model.layers[k].precedents.outputs).T
                                        else:
                                                #il n'y pas pas de couche precedente, donc on peut directement mettre à jour W
                                                self.model.layers[k].W = self.model.layers[k].W - self.lr * np.matmul(gradient,x).T

                                                #gradient checking
                                                if self.checking:
                                                        print("diff", np.matmul(gradient,x).T-checking)
                                else:
                                        #Le calcul de la couche precedente a été fait, donc on met à jour W
                                        self.model.layers[k+1].W = self.model.layers[k+1].W - self.lr * gradientW
                
                #affiche le dernier gradient s'il est totalement nul, ou si on veut voir son evolution
                if self.suivi or not np.any(gradient):
                        #pass
                        #gradient de la premiere couche
                        #print("grad", gradient,self.model.layers[2].outputs)
                        print(gradient,"\n")


###########  Fin SGD  ###################



def multiclass_learning(nb_epoch, model, optimiseur, data):
    
        for epoch in range(nb_epoch):
                shuffle(data)
                for i in range(len(data)):

                        x,y =data[i]

                        #probabilites en sortie du RN, après softmax
                        prob_output=model.forward(x)                
                        #print(prob_output)
                        
                        #Perte calcule par la fonction de perte
                        #Negative log likelihood appliqué a vec_output                         
                        NLLL=-np.log(prob_output[0,y])
                        #print("NLLL", NLLL, NLLL.shape)

                        if (np.argmax(prob_output)!=y):

                                #gradient de la perte, -1/L(prob_output) à l'indice y est zero partout ailleurs  d'apres le calcul 
                                gradient_NLLL=np.zeros(prob_output.shape).T
                                gradient_NLLL[y,0]=-1/(prob_output[0,y])
                                #print("grad NLL", gradient_NLLL)


                                checkg=optimiseur.checking_gradient(y, x, prob_output)
                                #Optimisation pour descente de gradient stochastique et mise à jour des poids
                                optimiseur.step(gradient_NLLL, y, x,checkg)
                        
                        
                        


if __name__ == '__main__':

        tanh=Tanh()
        relu=Relu()
        softmax=Softmax()

        Xor=Network(2,2, tanh, softmax, 1, 50 )
        Opti=SGD(Xor,0.5)
        
        data=[ (np.array([-1, -1]).reshape((1,2)), 0) , (np.array([-1,1]).reshape((1,2)), 1),
               (np.array([1,-1]).reshape((1,2)), 1), (np.array([1, 1]).reshape((1,2)), 0) ]
        
        print("test avant apprentissage")
        for x,y in data:
                
                print(x)
                print("gold", y)
                probs=Xor.forward(x)
                #print("pred", np.argmax(probs))
                print("probs", probs)
                
                if (np.argmax(probs)!=y):
                        print(False)
                else:
                        print(True)
                print("---------------")
        print("fin test\n")

        multiclass_learning(1000, Xor, Opti, data )
        
        print("apres apprentissage")
        for x,y in data:
                
                print(x)
                print("gold", y)
                probs=Xor.forward(x)
                print("pred", np.argmax(probs))
                print("probs", probs)
                
                if (np.argmax(probs)!=y):
                        print(False)
                else:
                        print(True)
                        
                        
                print("---------------")
