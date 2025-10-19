# Plan de Implementación: Clips Incrementales en Memoria

## 1. Objetivo
- **[propósito]** Generar clips destacados post-sesión para oratoria incremental.
- **[alcance]** Capturar highlights (muletillas, pausas largas, buen ritmo, pérdida de contacto visual), generar videos cortos, subirlos a S3 y exponerlos vía API/UI.

## 2. Arquitectura General
- **[pipeline]**
  1. `TimelineGenerator` produce eventos con timestamps al finalizar la sesión incremental.
  2. `ClipScheduler` analiza esos eventos y encola trabajos en Redis (memoria).
  3. `clip_worker` procesa trabajos, recorta video/thumbnail con FFmpeg y sube a S3.
  4. Registra clips en `session_clips` (PostgreSQL) para consumo y limpieza de Redis.

## 3. Cambios Backend

### 3.1 Esquema de Datos
- **[PostgreSQL]**
  ```sql
  CREATE TABLE session_clips (
      id UUID PRIMARY KEY,
      session_id VARCHAR(255) REFERENCES sessions(id),
      event_type VARCHAR(50) NOT NULL,
      start_sec DECIMAL(10,4) NOT NULL,
      end_sec DECIMAL(10,4) NOT NULL,
      duration_sec DECIMAL(10,4) NOT NULL,
      s3_url TEXT NOT NULL,
      thumbnail_url TEXT,
      confidence DECIMAL(4,2),
      created_at TIMESTAMP DEFAULT NOW()
  );
  ```

- **[Redis]** Claves en memoria:
  - `clip_jobs:pending` (lista): IDs de trabajos pendientes.
  - `clip_job:{id}` (hash): `{session_id, event_type, start_sec, end_sec, status, attempts}`.
  - `clip_jobs:processing` (set opcional): seguimiento de trabajos en curso.

### 3.2 Generación de Timeline
- **[archivo]** `api/analysis/timeline_generator.py`
  - Extender para devolver eventos relevantes post-sesión: `filler`, `pause_long`, `good_rhythm`, `gaze_lost`.
  - Retornar estructura `{"events": [...], "duration": ...}`.

### 3.3 Scheduler de Clips
- **[archivo]** `api/services/clip_scheduler.py` (nuevo)
  ```python
  TARGET_EVENTS = {"filler", "pause_long", "good_rhythm", "gaze_lost"}
  class ClipScheduler:
      @staticmethod
      async def enqueue_session(session_id: str, timeline: dict) -> None:
          for event in timeline["events"]:
              if event["type"] not in TARGET_EVENTS:
                  continue
              job_id = str(uuid.uuid4())
              payload = {
                  "session_id": session_id,
                  "event_type": event["type"],
                  "start_sec": event["start"],
                  "end_sec": event["end"],
                  "status": "pending",
                  "attempts": 0
              }
              await redis.hset(f"clip_job:{job_id}", mapping=payload)
              await redis.rpush("clip_jobs:pending", job_id)
  ```
  - Reglas: añadir margen ±0.5s, limitar a N clips por tipo (configurable).

### 3.4 Worker de Clips
- **[archivo]** `workers/incremental_clip_worker.py` (nuevo)
  - Loop:
    1. `job_id = redis.blpop("clip_jobs:pending")`.
    2. Marca `status=processing`, mueve a set `clip_jobs:processing`.
    3. Obtiene video completo (`SessionCoordinator.video_manager.export_full_video(session_id)`).
    4. Corre FFmpeg para recortar clip y generar thumbnail.
    5. Sube clip y thumbnail a S3 (`speechup-incremental-clips/{session_id}/`).
    6. Inserta fila en `session_clips` con URLs (presigned generadas en API).
    7. Borra hash Redis y saca de `clip_jobs:processing`.
  - Manejo de errores: incrementa `attempts`, reencola con backoff, alerta si excede límite.

### 3.5 Integración en Pipeline Incremental
- **[archivo]** `api/websockets/incremental.py`
  - En `handle_incremental_oratory_feedback()`, tras el insert final a base externa:
    ```python
    timeline = timeline_generator.generate(session)
    await ClipScheduler.enqueue_session(session_id, timeline)
    ```
- **[archivo]** `api/websockets/session_coordinator.py`
  - Asegurarse de exportar video completo al finalizar (`video_manager.save_session_video(session_id)`).

## 4. API y Frontend

### 4.1 Endpoints
- **[archivo]** `api/routes/clips.py` (nuevo)
  - `GET /api/v1/sessions/{session_id}/clips` → lista de clips persistidos.
  - Opcional `GET /api/v1/sessions/{session_id}/clips/status` → número de clips pendientes (lee Redis).

### 4.2 App React Native
- **[componentes]**
  - `services/api.ts`: método `fetchSessionClips(sessionId)`.
  - `components/ClipsList.tsx`: renderiza thumbnails y reproduce clip (URL presignada) dentro de `Video`.
  - `screens/Dashboard.tsx`: muestra sección "Clips destacados" una vez disponibles.

## 5. Infraestructura
- **[Redis]** Provisionar instancia (Elasticache o contenedor). Variables `REDIS_URL`, TTL opcional.
- **[S3]** Crear bucket `speechup-incremental-clips-{env}` con políticas privadas + lifecycle.
- **[Storage temporal]** Directorio `/tmp/sessions/{session_id}` para video full y clips hasta subirlos.
- **[Deployment worker]** Empaquetar `clip_worker` como servicio (systemd/ECS). Logs a CloudWatch.

## 6. Validación
- **[tests]**
  - Unit tests `timeline_generator`. 
  - Tests de scheduler (mock Redis) y worker (mock FFmpeg/S3).
  - Prueba E2E: sesión → timeline → clips en S3 → API retorna resultados.
- **[monitoreo]**
  - Métricas: `clips_generated_total`, `clip_generation_latency`, `clip_job_failures`.
  - Alarmas ante >=3 reintentos o jobs en `processing` > 10 min.

## 7. Roadmap de Tareas
- **[Fase 1]** Esquema datos + Redis + timeline completo.
- **[Fase 2]** Scheduler/worker + exportación video.
- **[Fase 3]** API REST + UI clips.
- **[Fase 4]** Testing, monitoreo y tuning (limitar cantidad de clips, calidad FFmpeg).

## 8. Consideraciones y Riesgos
- **[durabilidad]** Mientras los jobs estén en Redis, dependen de memoria → implementar requeue al reiniciar worker.
- **[peso videos]** Garantizar limpieza (`tmp` cleanup) para no agotar disco.
- **[latencia]** Objetivo < 60s entre `session_completed` y clip disponible.
- **[seguridad]** Usar presigned URLs temporales (<= 1h) y validar acceso por usuario.

## 9. Estado actual
- **[completado]** Scheduler Redis (`api/services/clip_scheduler.py`) y hook de encolado en `api/websockets/incremental.py`.
- **[completado]** Timeline básico (`api/analysis/timeline_generator.py`).
- **[completado]** Worker stub (`workers/incremental_clip_worker.py`).
- **[pendiente]** Implementar recorte FFmpeg + upload S3 en worker.
- **[pendiente]** Crear tabla `session_clips` y endpoints REST/React Native para consumir clips.
- **[pendiente]** Definir infraestructura Redis/S3 y pruebas E2E.

## 10. Requisitos de despliegue
- **[variables de entorno]**
  - `CLIP_BUCKET`: nombre del bucket S3 (default: `clips-bucket-speech-up`).
  - `CLIPS_API_URL`: URL del backend Java (default: `http://98.91.55.213:7070`).
  - `REDIS_URL`: conexión a Redis (default: `redis://localhost:6379/0`).
  - `AWS_ACCESS_KEY_ID` y `AWS_SECRET_ACCESS_KEY`: credenciales con permisos `s3:PutObject` y `s3:GetObject`.
  - **Nota**: Los valores de CLIP_BUCKET, CLIPS_API_URL y REDIS_URL ya están definidos como constantes en el código. Solo necesitas exportar las variables de entorno si quieres sobrescribir estos defaults.
- **[Redis]** Instancia accesible desde API y worker para encolar trabajos (`clip_jobs:pending`).
- **[S3]** Bucket con políticas para permitir la subida de `session-{sessionId}/clipId.mp4` y thumbnails.
- **[Backend Java]**
  - Tabla `session_clips` creada vía `LocalBootstrap`.
  - Endpoint `POST /clips` que recibe `ClipsInsertDto` (sin `clipId`) y retorna `{id, sessionId, eventType}`.
- **[Python worker]** Dependencias instaladas (`ffmpeg-python`, `httpx`, `boto3`, `asyncpg`). Ejecutar con los env vars anteriores.

## 11. Guía económica de infraestructura
- **[Redis (opción mínima)]**
  - Usar un contenedor Docker en la misma instancia del worker/API:
    ```bash
    docker run -d --name speechup-redis -p 6379:6379 redis:7-alpine
    ```
  - Alternativa sin Docker: instalar Redis (`apt install redis-server`) y habilitar la protección por contraseña (`requirepass`).
  - Para environments multi-host, considerar Lightsail (AWS) o EC2 t4g.micro + sec group restringido por IP.
- **[S3 económico]**
  - Crear bucket en región `us-east-1` (más barata generalmente).
  - Habilitar ciclo de vida para mover objetos >30 días a `STANDARD_IA` o `GLACIER` si aplica.
  - Política mínima para usuario IAM dedicado:
    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
          "Resource": "arn:aws:s3:::<CLIP_BUCKET>/session-*/*"
        }
      ]
    }
    ```
- **[Credenciales AWS low-cost]**
  - Crear usuario IAM con acceso programático únicamente.
  - Generar access key/secret key y almacenarlos en AWS Secrets Manager o variable de entorno en la instancia.
- **[Monitoreo de costos]**
  - Configurar AWS Budgets con alertas <$10 USD/mes.
  - Habilitar logging de S3 (opcional) para auditar accesos.

---
Última actualización: Oct 18, 2025.
