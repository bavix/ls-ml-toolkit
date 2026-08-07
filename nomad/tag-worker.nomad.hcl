# Nomad job для воркера обучения (src/ls_ml_toolkit/worker.py, Dockerfile.worker).
#
# Заполнить перед запуском: datacenter, S3-бакет и креды, теги образа. Отмечено ниже #FILL.
# Ресурсы посчитаны под 5600G (6 ядер/12 потоков, 32 ГБ) — не всё ядро, оставлен запас системе.

job "ihowmuch-tag-worker" {
  datacenters = ["dc1"] # FILL: имя вашего датацентра Nomad
  type        = "service"

  group "worker" {
    count = 1

    restart {
      # Обучение может упасть на конкретной задаче (плохой рецепт, OOM) — воркер сам пишет
      # ошибку в jobs/done/<id>.json и продолжает поллинг. Рестарт нужен только если упал сам
      # процесс воркера, не задача обучения.
      attempts = 3
      interval = "10m"
      delay    = "30s"
      mode     = "delay"
    }

    task "worker" {
      driver = "docker"

      config {
        image = "REQUIRED-registry/ihowmuch-tag-worker:latest" # FILL: куда запушили образ
      }

      # Секреты — через env, не в самой job-спеке в открытом виде. Если в кластере есть Nomad
      # Variables или Vault-интеграция, эти четыре значения оттуда, а не хардкодом здесь.
      env {
        LS_ML_S3_ACCESS_KEY_ID     = "REQUIRED" # FILL
        LS_ML_S3_SECRET_ACCESS_KEY = "REQUIRED" # FILL
        LS_ML_S3_DEFAULT_REGION    = "us-east-1" # FILL при необходимости
        LS_ML_S3_ENDPOINT          = ""          # FILL: пусто для AWS S3, иначе URL S3-совместимого хранилища
      }

      args = [
        "run",
        "--bucket", "REQUIRED-bucket-name", # FILL
        "--prefix", "jobs",
        "--poll-seconds", "30",
      ]

      resources {
        cpu    = 5000 # 5 из ~6 ядер (5600G), MHz-эквивалент по номадовской модели
        memory = 24000 # 24 из 32 ГБ, запас системе и другим задачам на этой машине
      }

      # Долгий полинг — не health-check в обычном смысле, но живость процесса стоит проверять.
      # Убрать, если в кластере нет service discovery для batch-воркеров.
      # service {
      #   name = "ihowmuch-tag-worker"
      #   tags = ["training-worker"]
      # }
    }
  }
}
