import yaml

str = """
thinking: |
    1. I have search results for "David Tao latest news 2024"
    2. There are 5 results from different sources: Wikipedia, Yahoo Style Singapore, DrinkHacker, MyDramaList, and Facebook/Barclays Center
    3. The main topic is about David Tao (the musician) with news about his 2024 activities including:
       - Soul Power II tour
       - Promise of new album in late 2024
       - Concert dates including Brooklyn on November 30
       - TV show appearances
    4. I need to create a highly-informative filename following the convention
    5. Filename components:
       - Topic: david-tao-latest-news-2024
       - Count: 5results
       - Key sources: wikipedia+yahoo+facebook
    6. Content should include:
       - Front matter metadata (query, date, result count)
       - Normalized sections for each result
       - Source blocks with URL, title, and content
action: file_save
reason: Valid search results provided with clear content about David Tao's 2024 news that can be saved in a structured, grep-able markdown format
file_name: david-tao-latest-news-2024-5results-wikipedia+yahoo+facebook.md
file_content: |
    ---
    query: David Tao latest news 2024
    search_date: 2024
    result_count: 5
    topic: David Tao musician news
    key_topics: Soul Power II tour, new album 2024, concerts, TV appearances
    sources: Wikipedia, Yahoo Style Singapore, DrinkHacker, MyDramaList, Facebook Barclays Center
    ---

    # David Tao Latest News 2024

    ## Search Query
    **Query:** I want latest news of David Tao
    **Search Term:** David Tao latest news 2024
    **Results:** 5 sources

    ## Summary
    David Tao, the renowned musician, is actively touring and creating new music in 2024. Key highlights include his "Soul Power II" concert tour across Taiwan and internationally, a promised new album release in late 2024, and various TV show appearances.

    ---

    ## Results

    ### Result 1: Wikipedia - David Tao
    **Source:** Wikipedia
    **URL:** https://en.wikipedia.org/wiki/David_Tao
    **Title:** David Tao
    **Relevance:** Concert tour information

    **Content:**
    In 2024, Tao returned to large-scale concert touring with the "Soul Power II" tour, announcing and staging dates across Taiwan and the region, including shows

    **Key Points:**
    - Soul Power II tour in 2024
    - Large-scale concert touring
    - Shows across Taiwan and region

    ---

    ### Result 2: Yahoo Style Singapore - New Album Promise
    **Source:** Yahoo Style Singapore
    **URL:** https://sg.style.yahoo.com/david-tao-promises-album-2024-035800563.html
    **Title:** David Tao promises new album in late 2024
    **Relevance:** Album release announcement

    **Content:**
    I made a promise at the Arena that I will release a new album in 2024!" He then stated that he will also be releasing a cover album next year,

    **Key Points:**
    - New album promised for late 2024
    - Cover album planned for following year
    - Announcement made at Arena concert

    ---

    ### Result 3: DrinkHacker - David Tao (Different Person)
    **Source:** DrinkHacker
    **URL:** https://www.drinkhacker.com/author/davidthomastao/?srsltid=AfmBOoryHGoMzmANtrnjO6A3l1ewwhbg7JdTnPLQMhxXrPNU0Kx-gs6H
    **Title:** David Tao
    **Relevance:** Different David Tao (bourbon/whiskey industry)

    **Content:**
    In late 2024, they entered the bourbon market by relaunching Bourbon de ... Their latest whiskey is the third in their single vintage bourbon series, a

    **Note:** This appears to be a different David Tao in the spirits/whiskey industry, not the musician.

    ---

    ### Result 4: MyDramaList - TV Appearances
    **Source:** MyDramaList
    **URL:** https://mydramalist.com/people/133977-david-tao
    **Title:** David Tao (陶喆)
    **Relevance:** Television show appearances

    **Content:**
    TV Show ; 2024, Hit Song add. Chinese TV Show, 2024, 12 eps. (Ep. 1) (Guest). 12. (Ep. 1). Guest. 8.2 ; 2024, The Treasured Voice Season 5 add. Chinese TV Show,

    **Key Points:**
    - Guest appearance on "Hit Song" (Episode 1)
    - Appearance on "The Treasured Voice Season 5"
    - Chinese TV shows in 2024

    ---

    ### Result 5: Facebook/Barclays Center - Brooklyn Concert
    **Source:** Facebook - Barclays Center
    **URL:** https://www.facebook.com/barclayscenter/posts/just-announced-david-tao-is-bringing-his-soul-power-tour-to-brooklyn-on-november/1228722932627987/
    **Title:** JUST ANNOUNCED 🎤 David Tao is bringing his Soul Power Tour to Brooklyn
    **Relevance:** US tour date announcement

    **Content:**
    JUST ANNOUNCED David Tao is bringing his Soul Power Tour to Brooklyn on November 30. Stay tuned for ticket information.

    **Key Points:**
    - Soul Power Tour coming to Brooklyn
    - Date: November 30, 2024
    - Venue: Barclays Center
    - Ticket information pending

    ---

    ## Source Index

    1. **Wikipedia** - https://en.wikipedia.org/wiki/David_Tao
    2. **Yahoo Style Singapore** - https://sg.style.yahoo.com/david-tao-promises-album-2024-035800563.html
    3. **DrinkHacker** - https://www.drinkhacker.com/author/davidthomastao/ (different person)
    4. **MyDramaList** - https://mydramalist.com/people/133977-david-tao
    5. **Facebook/Barclays Center** - https://www.facebook.com/barclayscenter/posts/.../1228722932627987/

    ---

    ## Grep Keywords
    david-tao, soul-power-tour, soul-power-ii, new-album-2024, concert-tour, brooklyn, barclays-center, november-30, taiwan, chinese-tv, hit-song, treasured-voice, musician, 陶喆
"""

res = yaml.safe_load(str)
print(res["file_name"])
print(res["file_content"])