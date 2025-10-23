from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Launch Chrome
driver = webdriver.Chrome()

# Open Google
driver.get("https://www.google.com")

# Wait for 3 seconds
time.sleep(3)

# Print the title of the page
print("Page title:", driver.title)

# Close browser
#driver.quit()
input("Press Enter to close...")
driver.quit()

