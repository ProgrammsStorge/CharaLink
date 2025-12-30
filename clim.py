import os
import requests
import colorama
import sys
import configparser
import shutil

colorama.init()
config=configparser.ConfigParser()
config.read("config.ini")
llm_config=config["Service"]
#print("Clim - CharaLink install manager. \n")

if sys.argv[1]=="install":
    print(f"Installing character {sys.argv[2]}")
    if not os.path.exists(f"characters\\{sys.argv[2]}"):
        print(colorama.Fore.YELLOW+"Fetching download url...",end=" ")
        response_url = requests.get(llm_config.get("url") + "/clim?search=" + sys.argv[2]).text
        print(colorama.Fore.GREEN+"done")
        print(colorama.Fore.YELLOW + "Downloading...", end=" ")
        response  = requests.get(response_url).content
        with open("character.zip", 'wb') as file:
            file.write(response)
        print(colorama.Fore.GREEN + "done")
        print(colorama.Fore.YELLOW + "Extracting...", end=" ")
        shutil.unpack_archive('character.zip', f'characters\\{sys.argv[2]}')
        os.remove("character.zip")
        print(colorama.Fore.GREEN + "done")
        print(colorama.Back.GREEN + colorama.Fore.WHITE + "Installation complete"+colorama.Back.RESET + colorama.Fore.RESET)
    else:
        print(
            colorama.Back.RED + colorama.Fore.WHITE + f"{sys.argv[2]} is installed" + colorama.Back.RESET + colorama.Fore.RESET)
if sys.argv[1]=="uninstall":
    print(f"Uninstalling character {sys.argv[2]}")
    if os.path.exists(f"characters\\{sys.argv[2]}"):
        print(colorama.Fore.YELLOW+f"Uninstalling {sys.argv[2]}...",end=" ")
        shutil.rmtree(f"characters\\{sys.argv[2]}")
        print(colorama.Fore.GREEN + "done")
        print(
            colorama.Back.GREEN + colorama.Fore.WHITE + "Uninstallation complete" + colorama.Back.RESET + colorama.Fore.RESET)
    else:
        print(
            colorama.Back.RED + colorama.Fore.WHITE + f"{sys.argv[2]} is not installed" + colorama.Back.RESET + colorama.Fore.RESET)


if sys.argv[1]=="list":
    response = requests.get(llm_config.get("url") + "/clim").json()
    for i in response:
        print(colorama.Fore.GREEN+i["name"],"-","id: ",i["id"],"-","description: ",i["description"].replace("\n"," "))