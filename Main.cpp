/**
  Base sur les fichiers fournis pour les TP d'INF2705 a Polytechnique
  */

#include <iostream>

#include "BasicTimer.h"
#include "inf2705.h"
#include "MassSpringSystem.h"

// variables pour l'utilisation des nuanceurs
ProgNuanceur progBase;  // le programme de nuanceurs de base
GLint locVertex;
GLint locColor;

// matrices de du pipeline graphique
MatricePipeline matrModel;
MatricePipeline matrVisu;
MatricePipeline matrProj;

// les formes
FormeCube *cubeFil = NULL;

// variables pour définir le point de vue
double thetaCam = 0.0;        // angle de rotation de la caméra (coord. sphériques)
double phiCam = 0.0;          // angle de rotation de la caméra (coord. sphériques)
double distCam = 0.0;         // distance (coord. sphériques)

// variables d'état
bool enmouvement = false;     // le modèle est en mouvement automatique ou non
bool afficheAxes = true;      // indique si on affiche les axes

const double dimBoite = 0.25; // la dimension de la boite
const glm::vec3 FocusPoint(0.f, 0.f, 0.1f);

const GLdouble thetaInit = 270.0;
const GLdouble phiInit = 80.0;
const GLdouble distInit = 0.45;

MassSpringSystem* MSS = NULL;
uint MSSnX = 2;
uint MSSnY = 2;
uint MSSnZ = 2;
glm::vec3 MSSsize(0.1f, 0.1f, 0.1f);

bool bStepOnce = false;

// vérifier que les angles ne débordent pas les valeurs permises
void verifierAngles()
{
   if ( thetaCam > 360.0 )
      thetaCam -= 360.0;
   else if ( thetaCam < 0.0 )
      thetaCam += 360.0;

   const GLdouble MINPHI = 0.01, MAXPHI = 180.0 - 0.01;
   if ( phiCam > MAXPHI )
      phiCam = MAXPHI;
   else if ( phiCam < MINPHI )
      phiCam = MINPHI;
}

void ReInitMSS()
{
    if (MSS) { delete MSS; MSS = NULL; }
    MSS = new MassSpringSystem();
    MSS->GenerateCube(MSSnX, MSSnY, MSSnZ, MSSsize);
    MSS->InitGL(locVertex, locColor);
    MSS->InitCuda();
}

void calculerPhysique( float DeltaT )
{
   if ( enmouvement || bStepOnce )
   {
       DeltaT = 0.01667f;
       //std::cout << "DeltaT = " << DeltaT << std::endl;
       const int NumIter = 18;
       for (int i = 0; i < NumIter; i++)
       {
           //ComputeForces();
           //IntegrateParticles(DeltaT / NumIter);
           MSS->UpdateSystem(DeltaT / NumIter);
           if (bStepOnce)
           {
               bStepOnce = false;
               break;
           }
       }
   }
}

void chargerNuanceurs()
{
   // charger le nuanceur de base
   progBase.creer( );
   progBase.attacher( GL_VERTEX_SHADER, "nuanceurSommetsBase.glsl" );
   progBase.attacher( GL_FRAGMENT_SHADER, "nuanceurFragmentsBase.glsl" );
   progBase.lier( );
   locVertex = progBase.obtenirAttribLocation( "Vertex" );
   locColor = progBase.obtenirAttribLocation( "Color" );
}


void initialiser()
{
   // positionnement de la caméra: angle et distance de la caméra à la base du bras
   thetaCam = thetaInit;
   phiCam = phiInit;
   distCam = distInit;

   // donner la couleur de fond
   glClearColor( 0.0, 0.0, 0.0, 1.0 );

   // activer les etats openGL
   glEnable( GL_DEPTH_TEST );

   // charger les nuanceurs
   chargerNuanceurs();

   // créer quelques autres formes
   progBase.utiliser( );
   cubeFil = new FormeCube( 1.0, false );

   ReInitMSS();
}

void conclure()
{
   delete cubeFil;
}

void afficherSysteme()
{
   matrModel.PushMatrix();{ // sauvegarder la tranformation courante

      progBase.assignerUniformMatrix4fv( "matrModel", matrModel );

      MSS->Draw();

   }matrModel.PopMatrix(); // revenir à la transformation sauvegardée
}

void definirCamera()
{
    matrVisu.LookAt( distCam*cos(glm::radians(thetaCam))*sin(glm::radians(phiCam)) + FocusPoint.x,
                     distCam*sin(glm::radians(thetaCam))*sin(glm::radians(phiCam)) + FocusPoint.y,
                     distCam*cos(glm::radians(phiCam)) + FocusPoint.z,
                     FocusPoint.x, FocusPoint.y, FocusPoint.z,
                     0., 0., 6. );
}

void FenetreTP::afficherScene()
{
   // effacer l'ecran et le tampon de profondeur
   glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

   progBase.utiliser( );

   // définir le pipeline graphique
   matrProj.Perspective( 45.0, (GLdouble) largeur_ / (GLdouble) hauteur_, 0.05, 300.0 );
   progBase.assignerUniformMatrix4fv( "matrProj", matrProj ); // informer la carte graphique des changements faits à la matrice

   definirCamera();
   progBase.assignerUniformMatrix4fv( "matrVisu", matrVisu ); // informer la carte graphique des changements faits à la matrice

   matrModel.LoadIdentity();
   progBase.assignerUniformMatrix4fv( "matrModel", matrModel ); // informer la carte graphique des changements faits à la matrice

   // afficher les axes
   if ( afficheAxes ) FenetreTP::afficherAxes();

   // tracer la boite englobante
   glVertexAttrib3f( locColor, 1.0, 0.5, 0.5 ); // équivalent au glColor() de OpenGL 2.x
   matrModel.PushMatrix();{
      matrModel.Translate( 0, 0, dimBoite/2 );
      matrModel.Scale( dimBoite, dimBoite, dimBoite );
      progBase.assignerUniformMatrix4fv( "matrModel", matrModel );
      cubeFil->afficher();
   }matrModel.PopMatrix();

   // tracer la bestiole à pattes
   afficherSysteme();
}

void FenetreTP::redimensionner( GLsizei w, GLsizei h )
{
   glViewport( 0, 0, w, h );
}

void FenetreTP::clavier( TP_touche touche )
{
    int tmpSize = 2;

   switch ( touche )
   {
   case TP_ECHAP:
   case TP_q: // Quitter l'application
       if (MSS) { delete MSS; MSS = NULL; }
      quit();
      break;

   case TP_x: // Activer/désactiver l'affichage des axes
      afficheAxes = !afficheAxes;
      std::cout << "// Affichage des axes ? " << ( afficheAxes ? "OUI" : "NON" ) << std::endl;
      break;

   case TP_v: // Recharger les fichiers des nuanceurs et recréer le programme
      chargerNuanceurs();
      std::cout << "// Recharger nuanceurs" << std::endl;
      break;

   case TP_i: // Réinitiliaser le point de vue
      phiCam = phiInit; thetaCam = thetaInit; distCam = distInit;
      break;

   case TP_SOULIGNE:
   case TP_MOINS: // Reculer la caméra
      distCam += 0.2;
      break;
   case TP_PLUS: // Avancer la caméra
   case TP_EGAL:
      if ( distCam > 1.0 )
         distCam -= 0.2;
      break;

   case 'a':
   case TP_ESPACE: // Mettre en pause ou reprendre l'animation
      enmouvement = !enmouvement;
      break;

   case 's':
       bStepOnce = true;
       break;

   case '1':
   case '2':
   case '3':
   case '4':
   case '5':
   case '6':
   case '7':
   case '8':
   case '9':
       tmpSize = (touche - '0') * 2;
       if (tmpSize > 5) tmpSize *= 1.5f;
       MSSnX = MSSnY = MSSnZ = tmpSize;
       // Fall through
   case 'r':
       ReInitMSS();
       break;

   case 't':
       MSS->TogglePrintTime();
       break;

   default:
      std::cout << " touche inconnue : " << (char) touche << std::endl;
      break;
   }
}

int dernierX, dernierY;
static bool pressed = false;
void FenetreTP::sourisClic( int button, int state, int x, int y )
{
   // button est un parmi { TP_BOUTON_GAUCHE, TP_BOUTON_MILIEU, TP_BOUTON_DROIT }
   // state  est un parmi { TP_PRESSE, DL_RELEASED }
   pressed = ( state == TP_PRESSE );
   switch ( button )
   {
   case TP_BOUTON_GAUCHE: // Déplacer (modifier angles) la caméra
      dernierX = x;
      dernierY = y;
      break;
   }
}

void FenetreTP::sourisWheel( int x, int y )
{
   //const int sens = +1;
    //std::cout << "wheel " << x << ", " << y << std::endl;
    if (y > 0)
    {
        distCam -= 0.06f;
    }
    else
    {
        distCam += 0.06f;
    }
}

void FenetreTP::sourisMouvement( int x, int y )
{
   if ( pressed )
   {
      int dx = x - dernierX;
      int dy = y - dernierY;
      thetaCam -= dx / 3.0;
      phiCam   -= dy / 3.0;

      dernierX = x;
      dernierY = y;

      verifierAngles();
   }
}

int main( int argc, char *argv[] )
{
   // créer une fenêtre
   FenetreTP fenetre( "INF2705 TP" );

   // allouer des ressources et définir le contexte OpenGL
   initialiser();

   BasicTimer SimTimer;
   SimTimer.Start();

   bool boucler = true;
   while ( boucler )
   {
      // mettre à jour la physique
      calculerPhysique( SimTimer.GetTimeSinceLastCheck() );

      // affichage
      fenetre.afficherScene();
      fenetre.swap();

      // récupérer les événements et appeler la fonction de rappel
      boucler = fenetre.gererEvenement();
   }

   // détruire les ressources OpenGL allouées
   conclure();

   return 0;
}
