import time
import html2text
import os
import requests
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

class Browser:
    """
    A class to interact with a web browser using Selenium.
    """
    def __init__(self, coder, args):
        """
        Initializes the Browser instance.

        Args:
            coder: The Coder instance, used for accessing io and args.
            args: Command line arguments, used for headless option.
        """
        self.driver = None
        self.coder = coder
        self.io = coder.io  # Use io from the coder instance
        self.args = args    # Store args for headless option
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True
        self.setup()

    def setup(self):
        """
        Sets up the Selenium WebDriver.
        """
        if self.driver:
            return

        try:
            chrome_options = Options()
            if self.args.headless_browser:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu") # Recommended for headless
            chrome_options.add_argument("--no-sandbox") # Bypass OS security model, REQUIRED for Docker/sandboxed environments
            chrome_options.add_argument("--disable-dev-shm-usage") # overcome limited resource problems

            # Suppress webdriver-manager logs except errors
            import logging
            logging.getLogger('WDM').setLevel(logging.ERROR)

            # Use webdriver-manager to automatically download and manage the ChromeDriver
            service = ChromeService(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)

        except WebDriverException as e:
            self.io.tool_error(f"WebDriver setup failed: {e}")
            self.io.tool_error("Please ensure Chrome is installed and accessible.")
            self.driver = None # Ensure driver is None if setup fails
        except Exception as e:
            self.io.tool_error(f"An unexpected error occurred during WebDriver setup: {e}")
            self.driver = None # Ensure driver is None if setup fails


    def go_to_url(self, url):
        """
        Navigates to the specified URL.

        Args:
            url (str): The URL to navigate to.

        Returns:
            tuple: A tuple containing the page content summary (str) and interactive elements (list),
                   or (None, None) if navigation fails.
        """
        if not self.driver:
            self.io.tool_error("WebDriver is not initialized. Cannot navigate.")
            return None, None

        try:
            self.driver.get(url)
            time.sleep(2)  # Wait for the page to load (consider more robust waits later)
            return self.analyze_page()
        except WebDriverException as e:
            self.io.tool_error(f"Failed to navigate to {url}: {e}")
            return None, None
        except Exception as e:
            self.io.tool_error(f"An unexpected error occurred during navigation: {e}")
            return None, None

    def search_google(self, query):
        """
        Performs a Google search for the given query.

        Args:
            query (str): The search query.

        Returns:
            tuple: A tuple containing the page content summary (str) and interactive elements (list)
                   of the first search result page, or (None, None) if search fails.
        """
        if not self.driver:
            self.io.tool_error("WebDriver is not initialized. Cannot perform search.")
            return None, None

        # Extract API keys from the api_key list in args/config first
        api_key = None
        cse_id = None
        if self.args.api_key and isinstance(self.args.api_key, list):
            for item in self.args.api_key:
                if isinstance(item, str) and "=" in item:
                    provider, key = item.split("=", 1)
                    if provider == "google-api-key":
                        api_key = key
                    elif provider == "google-cse-id":
                        cse_id = key

        # Fallback to environment variables if not found in config list
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not cse_id:
            cse_id = os.environ.get("GOOGLE_CSE_ID")


        if api_key and cse_id:
            # Use API if keys are available and not empty strings
            return self._search_google_api(query, api_key, cse_id)
        else:
            # Fallback to Selenium with a warning
            self.io.tool_warning(
                "Google API Key/CSE ID not found in config file or environment variables."
                " Falling back to Selenium for search, which may be blocked by CAPTCHA."
                " Consider setting google-api-key and google-cse-id in aider.conf.yml"
                " or environment variables for reliable results."
            )
            try:
                search_url = f"https://www.google.com/search?q={query}"
                self.driver.get(search_url)
                time.sleep(2) # Wait for search results

                # Analyze the search results page itself.
                return self.analyze_page()

            except WebDriverException as e:
                self.io.tool_error(f"Google search via Selenium failed: {e}")
                return None, None
            except Exception as e:
                self.io.tool_error(f"An unexpected error occurred during Google search via Selenium: {e}")
                return None, None

    def _search_google_api(self, query, api_key, cse_id):
        """
        Performs a Google search using the Custom Search JSON API.

        Args:
            query (str): The search query.
            api_key (str): The Google API Key.
            cse_id (str): The Google Custom Search Engine ID.


        Returns:
            tuple: A tuple containing the formatted search results (str) and a list of links (list),
                   or (None, None) if the API call fails.
        """
        # Keys are passed in now, basic check
        if not api_key or not cse_id:
            # This check might be redundant if search_google always passes valid keys
            self.io.tool_error("Google API Key or CSE ID missing.")
            return None, None

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': api_key,
            'cx': cse_id,
            'q': query,
            'num': 5 # Request top 5 results
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            search_results = response.json()

            formatted_results = f"Search results for '{query}':\n\n"
            links_list = []

            if 'items' in search_results:
                for i, item in enumerate(search_results['items']):
                    title = item.get('title', 'No Title')
                    link = item.get('link', '#')
                    snippet = item.get('snippet', 'No Snippet').replace('\n', ' ')

                    formatted_results += f"{i+1}. {title}\n"
                    formatted_results += f"   Link: {link}\n"
                    formatted_results += f"   Snippet: {snippet}\n\n"
                    links_list.append(f"Link: {title} ({link})")
            else:
                formatted_results += "No results found."

            # Return the formatted string and the list of links
            # Note: The concept of "interactive elements" is less direct with API results,
            # so we return the links found.
            return formatted_results, links_list

        except requests.exceptions.RequestException as e:
            self.io.tool_error(f"Google API search failed: {e}")
            # Attempt to parse error details from Google's response if possible
            try:
                error_details = response.json()
                self.io.tool_error(f"API Error details: {json.dumps(error_details)}")
            except (AttributeError, ValueError, json.JSONDecodeError):
                 pass # Ignore if response is not available or not JSON
            return None, None
        except Exception as e:
            self.io.tool_error(f"An unexpected error occurred during Google API search: {e}")
            return None, None


    def analyze_page(self):
        """
        Analyzes the current page content and extracts interactive elements using Selenium.

        Returns:
            tuple: A tuple containing the page content summary (str) and interactive elements (list),
                   or (None, None) if analysis fails.
        """
        if not self.driver:
            self.io.tool_error("WebDriver is not initialized. Cannot analyze page.")
            return None, None

        try:
            # Get page source and convert to markdown
            page_source = self.driver.page_source
            text_content = self.html_converter.handle(page_source)

            # Limit content size (e.g., first 2000 characters) for summary
            summary_content = text_content[:2000]
            if len(text_content) > 2000:
                summary_content += "\n... (content truncated)"

            # Find interactive elements (buttons, links, inputs)
            interactive_elements_data = []
            try:
                # Find buttons
                buttons = self.driver.find_elements(By.TAG_NAME, "button")
                for btn in buttons:
                    text = btn.text or btn.get_attribute('aria-label') or btn.get_attribute('value') or "Unnamed Button"
                    interactive_elements_data.append(f"Button: {text.strip()}")

                # Find links
                links = self.driver.find_elements(By.TAG_NAME, "a")
                for link in links:
                    text = link.text or link.get_attribute('aria-label') or "Unnamed Link"
                    href = link.get_attribute('href')
                    if text.strip() and href: # Only include links with text and href
                         interactive_elements_data.append(f"Link: {text.strip()} ({href})")

                # Find inputs (text, submit, etc.)
                inputs = self.driver.find_elements(By.TAG_NAME, "input")
                for inp in inputs:
                    input_type = inp.get_attribute('type') or "text"
                    name = inp.get_attribute('name') or inp.get_attribute('aria-label') or inp.get_attribute('id') or f"Unnamed {input_type} input"
                    if input_type not in ["hidden"]: # Exclude hidden inputs
                        interactive_elements_data.append(f"Input ({input_type}): {name.strip()}")

                # Find textareas
                textareas = self.driver.find_elements(By.TAG_NAME, "textarea")
                for ta in textareas:
                    name = ta.get_attribute('name') or ta.get_attribute('aria-label') or ta.get_attribute('id') or "Unnamed textarea"
                    interactive_elements_data.append(f"Textarea: {name.strip()}")

            except NoSuchElementException:
                # It's okay if some element types are not found
                pass
            except WebDriverException as e:
                self.io.tool_warning(f"Could not completely analyze interactive elements: {e}")


            # TODO: Integrate with AI for summarization/analysis later
            # For now, just return the extracted text and elements
            return summary_content, interactive_elements_data

        except WebDriverException as e:
            self.io.tool_error(f"Failed to analyze page: {e}")
            return None, None
        except Exception as e:
            self.io.tool_error(f"An unexpected error occurred during page analysis: {e}")
            return None, None

    def quit(self):
        """
        Closes the browser and quits the WebDriver session.
        """
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
            except WebDriverException as e:
                self.io.tool_error(f"Error quitting browser: {e}")
            except Exception as e:
                self.io.tool_error(f"An unexpected error occurred while quitting browser: {e}")
        else:
            self.io.tool_output("Browser not running.")
