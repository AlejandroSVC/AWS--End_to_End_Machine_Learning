# Cloud - XGBoost distribuido en AWS Sagemaker usando PySpark

## Descripción general

Este código crea un flujo de trabajo completo para entrenar e implementar un modelo de clasificación binaria con XGBoost en AWS SageMaker, aprovechando PySpark para la lectura y procesamiento de big data. Está diseñado para computación distribuida e incluye funciones de seguridad y monitoreo. Esta implementación proporciona una canalización (pipeline) integral (end-to-end) de nivel de producción para el modelo de clasificación que escala según el tamaño de la base de datos.

El código se divide en siete pasos principales, cada uno con tareas específicas:

1) Configuración e instalación: Configura las credenciales de AWS, las rutas de S3 y la sesión de SageMaker, utilizando variables de entorno para la seguridad.
2) Procesamiento de datos con PySpark: Carga y preprocesa los datos, los divide en conjuntos de entrenamiento y prueba, y guarda los resultados con control de versiones para facilitar su trazabilidad.
3) Preparación del entrenamiento de XGBoost: Configura el entrenamiento con depuración, creación de perfiles y seguimiento de experimentos para una mejor monitorización.
4) Inicio del entrenamiento: Ejecuta el trabajo de entrenamiento con gestión de errores para mayor robustez.
5) Implementación del modelo: Implementa el modelo en un punto final con captura de datos para su monitorización.
6) Evaluación: Evalúa el rendimiento del modelo mediante la puntuación AUC.
7) Limpieza: Limpia recursos para evitar costes innecesarios.

## Clasificación Binaria con XGBoost Distribuido en AWS SageMaker usando PySpark

## Paso 1: Configuración y Preparación

Configurar un entorno seguro y aislado. Esta sección inicializa el entorno para el flujo de trabajo de aprendizaje automático, centrándose en la seguridad y organización:

Importar bibliotecas necesarias: SageMaker para flujos de trabajo de aprendizaje automático, Boto3 para interacciones con AWS y OS para variables de entorno.

```
import sagemaker
import boto3
import os
```
1.1. Usar variables de entorno para información sensible (Mejor Práctica)

Obtener el nombre del bucket S3 de las variables de entorno por seguridad; usar 'your-s3-bucket' como valor predeterminado si no está configurado. Obtener la región AWS de la sesión de Boto3, asegurando una configuración dinámica.

```
BUCKET = os.environ.get('SAGEMAKER_BUCKET', 'your-s3-bucket')
REGION = boto3.Session().region_name                                                   # Actual región AWS
```
1.2. Prefijos S3 para datos con control de versiones y aislamiento

Definir el nombre del proyecto y los prefijos S3 para datos y salida, organizando y aislando recursos, facilitando el control de versiones y la separación de responsabilidades.

```
PROJECT_NAME  = "xgboost-binary-classification"
DATA_PREFIX   = f"{PROJECT_NAME}/data"
OUTPUT_PREFIX = f"{PROJECT_NAME}/output"
```
Construir rutas S3 para datos de entrada en formato parquet, conjuntos de entrenamiento y prueba, y artefactos de salida, asegurando que todas las interacciones con datos estén versionadas y aisladas.

Rutas S3 para datos y salidas:
```
PARQUET_PATH = 	f's3://{BUCKET}/{DATA_PREFIX}/parquet/'    # Ubicación de datos de entrada
TRAIN_PREFIX = 	f's3://{BUCKET}/{DATA_PREFIX}/train/'      # Salida de datos de entrenamiento
TEST_PREFIX = 	f's3://{BUCKET}/{DATA_PREFIX}/test/'       # Salida de datos de testeo
OUTPUT_PATH = 	f's3://{BUCKET}/{OUTPUT_PREFIX}/'          # Artefactos del modelo
```
1.3. Inicializar sesión de SageMaker y obtener rol de ejecución

Esta sección asegura una configuración segura y organizada, usando variables de entorno para información sensible como nombres de buckets S3, alineándose con las mejores prácticas de seguridad de AWS.

Inicializar sesión de SageMaker para gestionar interacciones con los servicios de SageMaker. Obtener el rol de ejecución para permisos, crucial para el control de acceso en AWS. Crear un cliente Boto3 para SageMaker para realizar llamadas API, permitiendo la gestión programática.

```
sess = sagemaker.Session()
role = sagemaker.get_execution_role()     # rol IAM para SageMaker
sm_client = boto3.client("sagemaker")     # cliente SageMaker
```
Definir el nombre de la columna objetivo para clasificación, una práctica estándar en tareas de clasificación binaria. Inicializar una lista vacía para características numéricas; se inferirán más adelante para adaptarse a esquemas de datos dinámicos.

```
TARGET_COL = 'target'
NUMERIC_FEATURES = []      # Completar si son conocidos. Si se deja vacío, se autocompletará
```
Mejores Prácticas:
Usar nombres de buckets y prefijos parametrizados para facilitar la reutilización.
Configurar carpetas S3 separadas para entrada, salida y artefactos del modelo.
Evitar codificar secretos e información sensible en los scripts.
                   
## Paso 2: Procesamiento de Datos con PySpark

Este paso maneja la carga, preprocesamiento y división de datos, aprovechando PySpark para el procesamiento distribuido. Esta sección realiza un procesamiento robusto de datos, con registro y control de versiones para garantizar trazabilidad y monitoreo, crucial para flujos de trabajo a gran escala.
                   
2.1. Inicializar Sesión de Spark y Usar Registro

Importar SparkSession para procesamiento de datos y logging para monitoreo, esencial para el manejo distribuido de datos y depuración.
```
from pyspark.sql import SparkSession
import logging
```
Crear una sesión de Spark con registro de eventos habilitado para almacenar logs en S3, facilitando el diagnóstico y análisis de rendimiento.
```
spark = SparkSession.builder \
    .appName("XGBoostParquetProcessing") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", f"s3://{BUCKET}/{PROJECT_NAME}/spark-logs/") \     # logueo S3
    .getOrCreate()
```
Habilitar registro de eventos de Spark para diagnóstico (Mejor Práctica)
Configurar el registro a nivel INFO para capturar eventos importantes y errores, asegurando visibilidad en el pipeline de procesamiento de datos.
```
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```
2.2. Leer Datos y Validar Esquema

Leer los datos parquet desde S3 en un DataFrame de Spark, un formato optimizado para big data.
Imprimir el esquema para verificar tipos de datos y asegurar compatibilidad. Registrar el número de filas cargadas para seguimiento del volumen de datos.
```
df = spark.read.parquet(PARQUET_PATH)
df.printSchema()                                               # Inspeccionar el esquema
logger.info(f"Loaded {df.count()} rows from Parquet.")
```
Validación de datos (Mejor Práctica)
```
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in data!")
```
Verificar si la columna objetivo existe en el DataFrame; lanzar un error si no está para garantizar integridad de los datos, crítico para el entrenamiento del modelo.

Recomendación: Validar el esquema de datos temprano y registrar conteos de filas.

2.3. Preprocesamiento de Datos con Monitoreo

Detectar automáticamente características numéricas si no están predefinidas
```
from pyspark.sql.functions import col
```
Importar la función col de PySpark para operaciones de columnas, necesaria para la manipulación de datos.

Convertir características no numéricas en el dataframe a numéricas
```
if not NUMERIC_FEATURES:
    NUMERIC_FEATURES = [f.name for f in df.schema.fields
                        if str(f.dataType) in ('DoubleType', 'FloatType', 'IntegerType', 'LongType') and f.name != TARGET_COL]
```
Si no se especifica características numéricas, inferirlas seleccionando columnas con tipos de datos numéricos, excluyendo la columna objetivo, para asegurar que solo se usen características relevantes.
```
df = df.select(NUMERIC_FEATURES + [TARGET_COL])    # Seleccionar sólo las columnas relevantes 
```
Seleccionar sólo las características numéricas y la columna objetivo para el DataFrame, reduciendo dimensionalidad y enfocándose en datos relevantes.

Monitoreo básico de calidad de datos (Mejor Práctica): verificar valores faltantes
```
missing_counts = df.select([col(c).isNull().cast("int").alias(c) for c in df.columns]) \
    .groupBy().sum().collect()[0].asDict()
logger.info(f"Missing counts per column: {missing_counts}")
```
Calcular el número de valores faltantes en cada columna y registrar los resultados para monitorear la calidad de los datos, esencial para garantizar limpieza de datos.
```
df = df.na.fill(0)            # Rellenar valores faltantes con 0
```
Rellenar valores faltantes con 0 para manejar nulos en los datos, una estrategia común de imputación para datos numéricos, aunque pueden necesitarse ajustes específicos del dominio.

2.4. División de Datos con Salida Versionada

Dividir el DataFrame en conjuntos de entrenamiento y prueba con 80% y 20% respectivamente, usando una semilla para reproducibilidad, asegurando divisiones consistentes para evaluación del modelo.
```
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)    # Dividir datos (80% entrenamiento, 20% prueba)
```
Guardar DataFrames de entrenamiento y prueba con rutas versionadas para trazabilidad (Mejor Práctica)
La siguiente sección genera una cadena de versión basada en la marca de tiempo actual para trazabilidad.
```
import datetime
version = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
```
Guardar los DataFrames de entrenamiento y prueba en S3 en formato CSV con encabezados, usando rutas versionadas para evitar sobrescribir datos previos y asegurar reproducibilidad.
```
train_path = f"{TRAIN_PREFIX}{version}/"
test_path   = f"{TEST_PREFIX}{version}/"
```
Escribir conjuntos de datos en S3 como CSV
```
train_df.write.mode('overwrite').csv(train_path, header=True)
test_df.write.mode('overwrite').csv(test_path, header=True)
```
## Paso 3: Preparar Trabajo de Entrenamiento con XGBoost

3.1. Habilitar Depurador, Perfilador y Seguimiento de Experimentos

Especificar el tipo y conteo de instancias para entrenamiento. Aquí, se usan instancias ml.c5.4xlarge, adecuadas para conjuntos de datos entre 10-100GB, con 2 instancias para entrenamiento distribuido, optimizando costo y rendimiento.
```
from sagemaker.debugger import Rule, rule_configs, DebuggerHookConfig
from sagemaker.session import TrainingInput
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.experiments.run import Run, load_run
```
Importar módulos necesarios para depuración en SageMaker, entradas de entrenamiento, estimador XGBoost y seguimiento de experimentos, mejorando la visibilidad del desarrollo del modelo.

Configuración de infraestructura de entrenamiento
```
INSTANCE_TYPE = 'ml.c5.4xlarge'         # Para bases de datos de tamaño 10-100GB
INSTANCE_COUNT = 2                      # Entrenamiento distribuído
```
Habilitar Depurador y Perfilador
```
debugger_hook_config = DebuggerHookConfig(
    s3_output_path=f"s3://{BUCKET}/{PROJECT_NAME}/debugger/"
)
```
Reglas de calidad de entrenamiento
```
rules = [
    Rule.sagemaker(rule_configs.loss_not_decreasing()),       # Detectar entrenamiento estancado
    Rule.sagemaker(rule_configs.overfit())                    # Detectar sobreajuste
             ]
```
Configurar DebuggerHookConfig para guardar datos de depuración en S3, permitiendo análisis de métricas de entrenamiento.
Definir reglas para monitorear pérdida no decreciente y sobreajuste durante el entrenamiento, mejorando la confiabilidad del modelo.

Crear Experimento de SageMaker para seguimiento (Mejor Práctica)
```
from sagemaker.experiments.experiment import Experiment
experiment = Experiment.create(
    experiment_name=f"{PROJECT_NAME}-exp",
    description="XGBoost binary classification experiment",
    sagemaker_boto_client=sm_client
)
```
Crear un Experimento de SageMaker para organizar y rastrear ejecuciones de entrenamiento, proporcionando un nombre y descripción, facilitando la comparación y reproducibilidad de experimentos.

Dentro de un contexto de Ejecución para seguimiento de experimentos, inicializar un estimador XGBoost con versión de framework especificada, rol, configuraciones de instancia, ruta de salida, hiperparámetros para clasificación binaria, configuración de depurador, reglas y habilitar entrenamiento con instancias spot para eficiencia de costos. Establecer distribución a servidor de parámetros si se usan múltiples instancias, optimizando para entrenamiento distribuido.

Contexto de seguimiento de experimentos
```
with Run(experiment_name=experiment.experiment_name,
                  run_name=f"run-{version}",
                  sagemaker_session=sess) as run:

    # Configurar estimador XGBoost

    estimator = XGBoost(
        framework_version='1.7-1',              # versión de XGBoost
        role=role,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        output_path=OUTPUT_PATH,                # Ubicación de los artefactos del modelo
        hyperparameters={
            'objective': 'binary:logistic',     # Clasificación binaria
            'num_round': 100,                   # Rondas de entrenamiento
            'max_depth': 5,                     # Complejidad del árbol
            'eta': 0.2,                         # Tasa de aprendizaje
            'eval_metric': 'auc',               # Métrica de evaluación
            'tree_method': 'gpu_hist' if 'p3' in INSTANCE_TYPE else 'hist',   # Aceleración de la GPU
        },
        debugger_hook_config=debugger_hook_config,
        rules=rules,
        use_spot_instances=True,             # Instancias spot para ahorro de costos
        max_wait=7200,                       # Establecer tiempo máximo de espera esperado en 2x
        max_run=3600,                        # Establecer tiempo máximo de entrenamiento esperado
        # Configuración de entrenamiento distribuído 
        distribution={
            'parameter_server': {'enabled': True}
        } if INSTANCE_COUNT > 1 else None
    )
```
Esta sección asegura monitoreo exhaustivo y optimización de costos, con seguimiento de experimentos para desarrollo iterativo de modelos.

Mejores Prácticas: 

1) Habilitar Depurador y Perfilador para monitorear entrenamiento y detectar problemas. 
2) Usar Experimentos de SageMaker para rastrear ejecuciones y metadatos para reproducibilidad. 
3) Usar Entrenamiento con Instancias Spot para reducir costos (use_spot_instances=True). 

## Paso 4: Lanzar Entrenamiento con Validación de Entrada

Este paso ejecuta el trabajo de entrenamiento con manejo de errores:
Importar TrainingInput para especificar canales de entrada de datos, esencial para definir fuentes de datos para entrenamiento.
```
from sagemaker.inputs import TrainingInput
```
Crear objetos TrainingInput para datos de entrenamiento y prueba, especificando rutas S3 y tipo de contenido como CSV, asegurando compatibilidad con SageMaker.
```
train_input = TrainingInput(train_path, content_type='csv')
test_input   = TrainingInput(test_path, content_type='csv')
```
Iniciar trabajo de entrenamiento con conjunto de validación
Usar try/except para lanzamiento robusto de trabajos de entrenamiento (Mejor Práctica)
Lanzar el trabajo de entrenamiento con entradas de entrenamiento y validación, esperando completitud para asegurar ejecución. Registrar éxito o fallo, y lanzar cualquier excepción para manejo adicional, mejorando robustez.
```
try:
    estimator.fit({'train': train_input, 'validation': test_input}, 
                             wait=True)           # Bloquear hasta el fin
    logger.info("Training job completed successfully.")
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise
```
Este paso asegura que el proceso de entrenamiento sea confiable, con registro para solución de problemas.

## Paso 5: Despliegue del Modelo e Inferencia

Esta sección despliega el modelo y configura monitoreo:

Usar Control de Versiones de Modelo y Auto-Escalado de Puntos Finales
```
from sagemaker.model_monitor import DataCaptureConfig
```
Importar DataCaptureConfig para monitorear modelos desplegados, crucial para análisis post-despliegue.

Habilitar Captura de Datos para monitoreo (Mejor Práctica):
Configurar captura de datos para registrar 100% de solicitudes y respuestas de inferencia en S3 para monitoreo, asegurando visibilidad del rendimiento del modelo en producción.
```
data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,      # Capturar todas las inferencias
    destination_s3_uri=f"s3://{BUCKET}/{PROJECT_NAME}/datacapture/"
)
```
Desplegar el modelo entrenado en un punto final con una instancia ml.m5.large, especificando el nombre del punto final con versión para trazabilidad, y habilitar captura de datos para monitoreo.
```
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',                             # Instancia costo-efectiva
    endpoint_name=f"{PROJECT_NAME}-endpoint-{version}",
    data_capture_config=data_capture_config                  # Habilitar monitoreo
)
```
Crear una configuración de punto final con una variante de producción usando el modelo del último trabajo de entrenamiento. Nota: Este paso podría ser redundante ya que estimator.deploy ya maneja la creación del punto final, indicando posible redundancia en la configuración.
Esta sección asegura que el modelo se despliegue con monitoreo, aunque la llamada adicional a la configuración del punto final puede ser innecesaria.
```
from sagemaker import Predictor

sm_client.create_endpoint_config(                         # Crear configuración de punto final
    EndpointConfigName=f"{PROJECT_NAME}-endpoint-config-{version}",
    ProductionVariants=[{
        'VariantName': 'AllTraffic',
        'ModelName': estimator.latest_training_job.name,  # Modelo entrenado
        'InitialInstanceCount': 1,
        'InstanceType': 'ml.m5.large'
    }]
)
```
## Paso 6: Evaluación del Modelo

Esta sección evalúa el modelo.
Importar pandas para manipulación de datos y roc_auc_score para evaluación, herramientas estándar para evaluación de modelos.
```
import pandas as pd
from sklearn.metrics import roc_auc_score
```
Descargar una muestra de datos de prueba si es necesario

Leer un archivo CSV local de muestra de prueba, extraer características y etiquetas verdaderas. Nota: Esto asume que la muestra de prueba está disponible localmente; en la práctica, podría necesitar descargarse de S3 o usarse los datos de prueba guardados previamente, indicando una posible dependencia de archivos locales.

Cargar muestra local de prueba para evaluación.
```
test_sample = pd.read_csv('local_test_sample.csv')
X = test_sample[NUMERIC_FEATURES]
y_true = test_sample[TARGET_COL]
```
Inferencia por lotes para conjuntos grandes (Mejor Práctica).

Usar el predictor desplegado para obtener predicciones en las características de la muestra de prueba, adecuado para procesamiento por lotes de grandes conjuntos de datos.

Obtener predicciones del punto final:
```
preds = predictor.predict(X.values)
```
Calcular métrica de evaluación (puntuación AUC) usando etiquetas verdaderas y predicciones, e imprimirla, proporcionando una métrica para evaluación del rendimiento del modelo.
```
auc = roc_auc_score(y_true, preds)
print(f'AUC: {auc:.4f}')                             # Mostrar el desempeño del modelo
```
## Paso 7: Limpieza

Esta sección evalúa el modelo y limpia recursos:
Limpiar puntos finales y recursos no utilizados (Mejor Práctica)

Eliminar el punto final para evitar costos innecesarios, un paso crítico para la gestión de recursos en entornos en la nube.
```
predictor.delete_endpoint()          # Eliminar el punto final para evitar cargos
```
Opcionalmente, eliminar artefactos del modelo y datos S3 si no se necesitan


                   
