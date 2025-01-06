import os
import time
from urllib.parse import unquote, urlparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException
 
def setup_driver(download_dir):
    """ Set up Chrome driver with specific download preferences for handling PDFs automatically. """
    chrome_options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True  # Ensures PDFs download automatically
    }
    chrome_options.add_experimental_option("prefs", prefs)
    # chrome_driver_path = "/usr/local/bin/chromedriver"
    chrome_driver_path = "/Users/phuonglh/tools/chromedriver-mac-x64/chromedriver"
    chrome_options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    # pip install --upgrade urllib3==1.26.16 (to fix a bug)
    return webdriver.Chrome(executable_path=chrome_driver_path, options=chrome_options)
    

 
def login(driver, base_url, username, password):
    """ Log into the website using credentials. """
    driver.get(base_url + "/login/#")  # Adjust this URL based on the actual login page
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "username_field"))  # Adjust ID as necessary
    )
    driver.find_element(By.ID, "username_field").send_keys(username)  # Adjust ID as necessary
    driver.find_element(By.ID, "password_field").send_keys(password)  # Adjust ID as necessary
    driver.find_element(By.ID, "login_button").click()  # Adjust ID as necessary
 
def navigate_and_download(base_url, driver, download_dir):
    """ Navigate through pages and download items. """
    driver.get(base_url)
    while True:
        try:
            items = WebDriverWait(driver, 30).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a[style='color: black;']"))
            )
        except TimeoutException:
            print("Timeout while waiting for items to load.")
            break
 
        for item in items:
            item_href = item.get_attribute('href')
            if not item_href.startswith('http'):
                item_url = 'https://repository.vnu.edu.vn' + item_href
            else:
                item_url = item_href
 
            driver.execute_script("window.open(arguments[0]);", item_url)
            driver.switch_to.window(driver.window_handles[1])
 
            try:
                download_link = WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "a.view[target='_blank']"))
                )
                file_url = download_link.get_attribute('href')
                file_name = unquote(urlparse(file_url).path.split('/')[-1])
                file_path = os.path.join(download_dir, file_name)
 
                if not os.path.exists(file_path):
                    download_link.click()
                    time.sleep(2)
                else:
                    print(f"File already exists: {file_name}")
            except TimeoutException:
                print(f"Timeout while trying to download from {item_url}")
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
 
        try:
            next_page = WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.page-link.next"))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", next_page)
            driver.execute_script("arguments[0].click();", next_page)
        except NoSuchElementException:
            print("No more pages to navigate.")
            break
        except TimeoutException:
            print("Timeout waiting for the next page button.")
            break
 
def main():
    download_dir = "/tmp/Downloads"
    base_url = "https://repository.vnu.edu.vn/"
    # base_url = "https://repository.vnu.edu.vn/browse?type=title"
    # base_url = "https://repository.vnu.edu.vn/handle/VNU_123/33312"
    username = 'phuonglh'
    password = '???'
    driver = setup_driver(download_dir)
    try:
        login(driver, base_url, username, password)  # Perform login before navigating
        print("Login success")
        navigate_and_download(base_url, driver, download_dir)
    finally:
        driver.quit()
 
if __name__ == "__main__":
    main()

    