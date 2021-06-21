IMPORTANTE: ANTES DE EJECUTAR EL CÓDIGO se debe crear una carpeta llamada Case1
DENTRO DE LA CARPETA "separated" que contenga SOLAMENTE las carpetas "Asian" y "Black"
de la base de datos, ya que esta será la carpeta que leerá el programa para buscar imágenes
en el caso del problema de clasificación binario. Es importante NO COPIAR las carpetas
"Asian" y "Black" de "separated" a Case1, osino cuando el programa lea la carpeta "separated" para el
problema multiclase, recibirá dos veces las imagenes de las clases "Asian" y "Black".

Teniendo listo este paso, se puede ejecutar el archivo main.cpp y seguir los pasos que se solicitan:

1) Escribir "1" o "2" (sin las comillas) en la terminal para indicar si se desea evaluar el problema de clasificación binario
o multiclase (1 es para binario, 2 es para multiclase)

2) Escribir el directorio donde se encuentre la carpeta "separated" o "Case1" según corresponda
Ej: Si en el paso anterior se escribió "1", aquí se escribiría ~/Escritorio/miCodigo/separated/Case1

Luego, si todo fue ingresado correctamente, el código debería ejecutarse e imprimir en pantalla los
resultados de clasificación.
