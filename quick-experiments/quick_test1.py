import os
import ctypes
import sys

print("Проверка DLL для TensorFlow...")
print("=" * 60)

# Список необходимых DLL
required_dlls = [
    'msvcp140.dll',  # Visual C++ 2015-2022
    'msvcp140_1.dll',  # Visual C++ 2015-2022 (доп.)
    'vcruntime140.dll',  # Visual C++ 2015-2022
    'vcruntime140_1.dll',  # Visual C++ 2015-2022 (доп.)
    'msvcp120.dll',  # Visual C++ 2013
    'msvcp110.dll',  # Visual C++ 2012
    'msvcp100.dll',  # Visual C++ 2010
]

# Системные пути для поиска
search_paths = [
    r'C:\Windows\System32',
    r'C:\Windows\SysWOW64',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin',
    r'C:\Program Files\NVIDIA Corporation\NVSMI',
]

found = 0
missing = 0

for dll in required_dlls:
    try:
        # Пробуем загрузить DLL
        ctypes.WinDLL(dll)
        print(f"✅ {dll} - НАЙДЕН")
        found += 1
    except OSError:
        # Ищем вручную
        dll_found = False
        for path in search_paths:
            if '*' in path:
                import glob

                for expanded_path in glob.glob(path):
                    dll_path = os.path.join(expanded_path, dll)
                    if os.path.exists(dll_path):
                        print(f"✅ {dll} - найден в {expanded_path}")
                        dll_found = True
                        found += 1
                        break
            else:
                dll_path = os.path.join(path, dll)
                if os.path.exists(dll_path):
                    print(f"✅ {dll} - найден в {path}")
                    dll_found = True
                    found += 1
                    break

        if not dll_found:
            print(f"❌ {dll} - НЕ НАЙДЕН")
            missing += 1

print("=" * 60)
print(f"Найдено: {found}, Отсутствует: {missing}")
print(f"Процент готовности: {(found / len(required_dlls)) * 100:.1f}%")

if missing > 0:
    print("\nУстановите Visual C++ Redistributable:")
    print("1. https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("2. Перезагрузите компьютер")