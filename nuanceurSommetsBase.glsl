#version 410

uniform mat4 matrModel;
uniform mat4 matrVisu;
uniform mat4 matrProj;

layout(location=0) in vec3 Vertex;
layout(location=2) in vec4 Color;

out vec4 couleur;

void main( void )
{
   gl_Position = matrProj * matrVisu * matrModel * vec4(Vertex, 1.0f);

   couleur = Color;
   //couleur = vec4(1.0f);
}
