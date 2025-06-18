#  DEBUG

1. python3.11 -m venv venv  (Windows: python -m venv venv)
2. source venv/bin/activate  (Windows: \venv\Scripts\activate)
3. pip install -e .
4. python -m aider.main

## Global Install with pipx (macOS, Linux и Windows)

1. brew install python@3.11
2. python3 -m pip install --user pipx
3. python3 -m pipx ensurepath
4. pipx install --python python3.11 git+https://github.com/VitalyKuzmin/my_aider.git
5. pipx upgrade my_aider
6. pipx uninstall my_aider

## Update from origin Aider (upstream)

1. git fetch upstream
2. git checkout main
3. git merge upstream/main
3.  Альтернативный способ (fetch + merge):
    git checkout main
    git pull upstream main
4. git push origin main



## Создание глобального доступа вручную (Альтернативный метод, macOS/Linux)

Если вы хотите запускать вашу модифицированную версию `aider` из любого места без необходимости каждый раз активировать виртуальное окружение (`source .venv/bin/activate`), вы можете создать символическую ссылку на исполняемый файл `aider` (или `my_aider`, в зависимости от того, как он называется в `.venv/bin`) из директории, которая уже есть в вашем системном `PATH`. Стандартное место для пользовательских скриптов в macOS/Linux - это `/usr/local/bin`.

1.  **Убедитесь, что `/usr/local/bin` существует и находится в вашем `PATH`:**
    *   Проверить `PATH`: `echo $PATH`
    *   Если `/usr/local/bin` нет, его может потребоваться создать (`sudo mkdir -p /usr/local/bin`) и добавить в `PATH` (обычно через конфигурационные файлы вашей оболочки, например, `~/.zshrc` или `~/.bash_profile`).

2.  **Найдите имя исполняемого файла:**
    Перейдите в корневую директорию вашего проекта `my_aider` и проверьте содержимое папки `.venv/bin`:
    ```bash
    ls .venv/bin
    ```
    Найдите исполняемый файл, который запускает вашу версию aider. После `pip install -e .` он должен называться `my_aider` (согласно `pyproject.toml`).

3.  **Создайте символическую ссылку:**
    Выполните эту команду из корневой директории вашего проекта `my_aider` (где находится папка `.venv`), используя имя `my_aider` для файла в `.venv/bin`:
    ```bash
    ln -s "$(pwd)/.venv/bin/my_aider" /usr/local/bin/my_aider
    ```
    *   `"$(pwd)/.venv/bin/my_aider"`: Получает полный путь к исполняемому файлу `my_aider` внутри вашего виртуального окружения.
    *   `/usr/local/bin/my_aider`: Создает ссылку с именем `my_aider` в `/usr/local/bin`.

    **Если вы получили ошибку `Permission denied`:** Вам может потребоваться выполнить команду с `sudo`:
    ```bash
    sudo ln -s "$(pwd)/.venv/bin/my_aider" /usr/local/bin/my_aider
    ```
    **Если вы получили ошибку `File exists`:** Это значит, что ссылка или файл с таким именем уже существует. Вы можете либо удалить существующий файл/ссылку (`sudo rm /usr/local/bin/my_aider`), либо использовать флаг `-f` (force) для перезаписи:
    ```bash
    sudo ln -sf "$(pwd)/.venv/bin/my_aider" /usr/local/bin/my_aider
    ```

4.  **Обновите кэш команд оболочки (для zsh и некоторых других):**
    Чтобы оболочка немедленно распознала новую команду, выполните:
    ```bash
    rehash
    ```

Теперь вы должны иметь возможность вызывать `my_aider` из любого терминала.

**Примечание:** Использование имени `my_aider` как для исполняемого файла, так и для ссылки, помогает избежать конфликтов с оригинальным `aider`, если он также установлен (например, через `pip install aider`).

*   **Windows:**
    В Windows создание символических ссылок таким образом менее распространено. Вместо этого рекомендуется добавить директорию со скриптами вашего виртуального окружения (`.venv\Scripts`) в системную переменную `PATH`.

    1.  **Найдите путь к директории `Scripts`:** В корне вашего проекта `my_aider` это будет путь вроде `C:\путь\к\вашему\проекту\my_aider\.venv\Scripts`. Скопируйте этот полный путь.
    2.  **Добавьте путь в `PATH`:**
        *   Нажмите `Win + R`, введите `sysdm.cpl` и нажмите Enter.
        *   Перейдите на вкладку "Дополнительно" и нажмите кнопку "Переменные среды...".
        *   В разделе "Системные переменные" (или "Переменные среды пользователя", если вы хотите установить только для себя) найдите переменную `Path` и нажмите "Изменить...".
        *   Нажмите "Создать" и вставьте скопированный путь к директории `.venv\Scripts`.
        *   Нажмите "ОК" во всех открытых окнах.
    3.  **Перезапустите терминал:** Откройте новое окно командной строки или PowerShell. Теперь вы должны иметь возможность вызывать `my_aider` из любого места.

