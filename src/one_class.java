
import com.mathworks.toolbox.javabuilder.*;
import java.io.*;
import Preprocesar_OC.*;
import Calcular_caracteristicas_oc.*;
import weka.clusterers.EM;
import weka.core.Instances;
import weka.core.converters.ArffLoader;


public class one_class 
{
	//Vector de características
	private final static String[] des_caracteristicas = {
			"columnas", "promedio", "dnormales", "sdt","std_menor","std_mayor", "num_cluster", "distancia_entre_cluster","media_entre_clusters", "maximo_entre_cluster", "mínimo_entre_cluster", "desviacion_entre_clusters", "distancias_intra_cluster", "media_intra_clusters", "maximo_intra_cluster", "minimo_intra_cluster","desviacion_intra_clusters"
	};
	
	//Funcion principal que llama a las funciones de preprocesamiento de datos, de cálculo de características, 
	//y de almacenamiento de resultados
	public static void main(String[] args) throws Exception {
	
		String path = "c:/datos_src/";  //fichero con las imagenes a tratar
		File directorio = new File(path);
		Object[] caracteristicas = null;
		String [] ficheros = directorio.list();
		

		//por cada imagen hago el preprocesado y el cálculo de características
		for (int i = 0; i < ficheros.length; i++) { 
			
			 if ( ficheros[i].substring(ficheros[i].length()-3, ficheros[i].length()).equals("bmp"))
			 {
			    Preprocesar_OC p = new Preprocesar_OC();  //Creo el objeto Preprocesar  
				Object[] results =null;
				System.out.println("El fichero: "+ficheros[i]);
				//divide en target, outliers, prepara conjunto de training y test y salva en arff
				results = p.preprocesa_oc(1, ficheros[i]);  
				
				//obtenemos el dataset preprocesado de la API matlab
				MWArray result = (MWNumericArray) results[0];    
				double[][] datos = (double[][]) result.toArray();
				
				//Creo el objeto calcular_características e invoco a la funcion de matlab
				//calcular_caracteristicas
				Calcular_caracteristicas_oc cc = new Calcular_caracteristicas_oc(); 
				caracteristicas = null;
				caracteristicas = cc.calcula_caracteristicas(1, datos); 
				
				//devuelvo el calculo de las características calculadas en matlab en un array (carac)
				MWArray caracteristica = (MWNumericArray) caracteristicas[0];
				double[][] carac = (double[][]) caracteristica.toArray();
				
				//obtengo la menor y tb. la mayor desviación típica de entre todos los 
				//atributos para almacenar como características, además, usaremos la 
				//menor como parámetro en el algoritmo EM
				double std_menor= 1000.0;
				double std_mayor = 0;
				for (int j=3;j < carac[0].length;j++)
				{
					if ( carac[0][j] < std_menor)
					{
						std_menor =  carac[0][j];
					}	
					if ( std_mayor < carac[0][j])
					{
						std_mayor = carac[0][j];
					}
				}
				
				//las características sobre el clustering se realizan sobre funciones sobre la API Weka
				EM cluster_EM = cluster_EM("c:/arff/"+ ficheros[i].substring(0, ficheros[i].length()-3)+ "arff",std_menor);
				int num_clusters = n_clusters(cluster_EM);
				
			    float[] distancias_entre_clusters = distancias_clusters(cluster_EM, carac[0][0]);
			  
			    //calcula las medidas sobre las distancias entre clusters
				float[] medidas_entre = calcula_media(distancias_entre_clusters);
			
				
				float[] distancias_intra_clusters = distancias_intra_clusters(cluster_EM, carac[0][0]);
				
				//calcula las medidas sobre las distancias intra clusters
				float [] medidas_intra = calcula_media(distancias_intra_clusters);
				
				//salvo en archivo los resultados de las características 
				salva_archivo(carac, std_menor, std_mayor, num_clusters, distancias_entre_clusters, medidas_entre, distancias_intra_clusters, medidas_intra ,ficheros[i].toString());
		
			}
		}
	}
	
	
	
	//calcula la media, max y min de un array de valores
	private static float[] calcula_media(float[] distancias) {
		float media =0;
		float max = -1;
		float min = 10000; //como los valores están normalizados, nos aseguramos que este valor 
		                   // realmente siempre sea un tope máximo
		float desviacion =0;
		
		for (int i=0; i < distancias.length; i++) {
			 media = media + distancias[i];
			 
			 if ( max < distancias[i])
				 max = distancias[i];
			 
			 if ( distancias[i] < min )
				 min = distancias[i];
			}

		media = media/distancias.length;
		
		for (int i=0; i < distancias.length; i++){
			
			desviacion += (distancias[i]-media)*(distancias[i]-media);	
		}
		
		desviacion = (float) Math.sqrt((desviacion/distancias.length));
		
		float [] datos = {media, max, min, desviacion};
			
		return datos;
	}

	
	
	
	//calcula las caraceristicas referentes al algoritmo EM invocando 
	//las clases de Weka
	public static EM cluster_EM(String fichero, double std_avg) throws Exception{
		
		double desviacion_tipica= std_avg;
		int num_iteraciones=100;
		
		//creo el algoritmo de cluster y configuro parámetros
		EM algoritmo_cluster = new EM();
		algoritmo_cluster.setMinStdDev(desviacion_tipica);
		algoritmo_cluster.setMaxIterations(num_iteraciones);
		
		//cargo el dataset desde arff
		ArffLoader loader = new ArffLoader();
		loader.setSource(new FileInputStream(fichero));
		Instances dataset = loader.getDataSet();
		
		//entrenar 
		algoritmo_cluster.buildClusterer(dataset);
		for (int i=0; i < dataset.numInstances(); i++)
			algoritmo_cluster.clusterInstance(dataset.instance(i));
	
		return algoritmo_cluster;	
	}
		
	
	
	//devuelve el número de clusters 
	public static int n_clusters(EM cluster_EM) throws Exception
	{
		return cluster_EM.numberOfClusters();
	}
	
	
	//devuelve la distancias entre clusters
	public static float[] distancias_clusters(EM algoritmo_cluster, double num_atributos) throws Exception
	{
		float[] num_distancias = new float[(int) (0.5 * algoritmo_cluster.numberOfClusters() * (algoritmo_cluster.numberOfClusters()-1))];
		float[][] valor_centroides = new float [algoritmo_cluster.numberOfClusters()][(int) num_atributos];
		
		//Almaceno en una matriz los valores de los centroides de cada cluster
		for (int m=0; m < algoritmo_cluster.numberOfClusters(); m++ )
		{
			for (int k=0; k < (int) num_atributos; k++)
			{
				//almaceno los valores de cada característica para cada cluster
				 valor_centroides[m][k] = (float) algoritmo_cluster.getClusterModelsNumericAtts()[m][k][0]; 		 
			}
		}
		
		//Para cada uno de los centroides calculo todas las distancias euclideas hacia los demás 
		//y los voy almacenando en el vector num_distancias
 		int distancias = 0;
		for (int i = 0; i< algoritmo_cluster.numberOfClusters()-1;i++)
		{
		    for (int j=i+1; j <= algoritmo_cluster.numberOfClusters()-1; j++)
		    {
		    	float temp=0;
		    	for (int k=0; k < (int) num_atributos;k++ )
		    	{
		    	//	System.out.print("los valores : " + valor_centroides[j][k] + " - " + valor_centroides[i][k] + "\n");
		    		temp += (valor_centroides[j][k] - valor_centroides[i][k]) * (valor_centroides[j][k] - valor_centroides[i][k]);
		    	}
		    	//System.out.println("La distancia " + distancias + " :" + Math.sqrt(temp));
		    	num_distancias[distancias]= (float) Math.sqrt(temp);
		    	distancias++;
		    }	
		}
		
		return num_distancias;
	}
	
	
	//devuelve la distancias intra clusters
	public static float[] distancias_intra_clusters(EM algoritmo_cluster, double num_atributos) throws Exception
	{
		float[] valor_distancias_intra = new float[algoritmo_cluster.numberOfClusters()];
		
		
		//Almaceno en una matriz los valores de las desviaciones típicas de cada atributo en cada cluster
		for (int m=0; m < algoritmo_cluster.numberOfClusters(); m++ )
		{
			for (int k=0; k < (int) num_atributos; k++)
			{
				  valor_distancias_intra[m] += algoritmo_cluster.getClusterModelsNumericAtts()[m][k][1];
				  
		    }
			//tomo como valor la media de las desviaciones típicas en cada cluster para todos sus atributos
			valor_distancias_intra[m]= (float) ((float) valor_distancias_intra[m] / num_atributos);
		}
		
		return valor_distancias_intra;
	}
	
	
	
	
	//Guarda los resultados de caracaterísticas de cada dataset
	public static void salva_archivo(double[][] carac, double std_menor, double std_mayor, int num_clusters, float[] distancias_entre_clusters,float[] medidas_entre, float[] distancias_intra_clusters, float[] medidas_intra, String fich) throws MWException {
		    
			try{

		      //Creamos un Nuevo objeto FileWriter dandole
		      //como parámetros la ruta y nombre del fichero
		      FileWriter fichero = new FileWriter("c:/resultados/"+(fich.substring(0,fich.length()-3))+"txt");

		   //insertamos las características en el archivo de texto de salida
		  	for (int j=0;j <=3 ;j++)
			 {
		        fichero.write(des_caracteristicas[j] + ": \r\n");
		        fichero.write("\t" + carac[0][j] + "\r\n");
			 }		
		  	//para las desviaciones de cada caracaterística
			for (int j=4;j < carac[0].length;j++)
			{
			    fichero.write("\t" + carac[0][j] + "\r\n");
			}
			
			fichero.write(des_caracteristicas[4] + ":\r\n ");
		  	fichero.write("\t" + std_menor + "\r\n");
		  
		  	fichero.write(des_caracteristicas[5] + ":\r\n ");
		  	fichero.write("\t" + std_mayor + "\r\n");
		  
			
		  	fichero.write(des_caracteristicas[6] + ":\r\n ");
		  	fichero.write("\t" + num_clusters + "\r\n");
		  	
		  	fichero.write(des_caracteristicas[7] + ": \r\n");
		    for (int j=0; j < distancias_entre_clusters.length;j++)
		    {
		      fichero.write("\t" + distancias_entre_clusters[j] + "\r\n");	   
		    }
		    
		    fichero.write(des_caracteristicas[8] + ": \r\n");
		    fichero.write("\t" + medidas_entre[0] + "\r\n");	   
		     
		    fichero.write(des_caracteristicas[9] + ": \r\n");
		    fichero.write("\t" + medidas_entre[1] + "\r\n");	   
		    
		    fichero.write(des_caracteristicas[10] + ": \r\n");
		    fichero.write("\t" + medidas_entre[2] + "\r\n");	   
		    
		    fichero.write(des_caracteristicas[11] + ": \r\n");
		    fichero.write("\t" + medidas_entre[3] + "\r\n");	   
		   
		    
		    fichero.write(des_caracteristicas[12] + ": \r\n");
		    
		    for (int j=0; j < distancias_intra_clusters.length;j++)
		    {
		      fichero.write("\t" + distancias_intra_clusters[j] + "\r\n");	   
		    }
		    
		    fichero.write(des_caracteristicas[13] + ": \r\n");
		    fichero.write("\t" + medidas_intra[0] + "\r\n");	   
		    
		    fichero.write(des_caracteristicas[14] + ": \r\n");
		    fichero.write("\t" + medidas_intra[1] + "\r\n");	   
		 
		    fichero.write(des_caracteristicas[15] + ": \r\n");
		    fichero.write("\t" + medidas_intra[2] + "\r\n");	   
		 
		    fichero.write(des_caracteristicas[16] + ": \r\n");
		    fichero.write("\t" + medidas_intra[3] + "\r\n");	   
		 
		    
		    //cerramos el fichero
		    fichero.close();

		    }catch(Exception ex){
		      ex.printStackTrace();
		    }
	}
	
}
