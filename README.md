# GPTCache עם מדיניות פינוי חכמה מבוססת-עלות (Cost-Aware Eviction)

[![Release](https://img.shields.io/pypi/v/gptcache?label=Release&color&logo=Python)](https://pypi.org/project/gptcache/)
[![pip download](https://img.shields.io/pypi/dm/gptcache.svg?color=bright-green&logo=Pypi)](https://pypi.org/project/gptcache/)
[![Codecov](https://img.shields.io/codecov/c/github/zilliztech/GPTCache/dev?label=Codecov&logo=codecov&token=E30WxqBeJJ)](https://codecov.io/gh/zilliztech/GPTCache)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit/)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/zilliz_universe.svg?style=social&label=Follow%20%40Zilliz)](https://twitter.com/zilliz_universe)
[![Discord](https://img.shields.io/discord/1092648432495251507?label=Discord&logo=discord)](https://discord.gg/Q8C6WEjSWV)

---

זוהי הרחבה של **[GPTCache](https://github.com/zilliztech/GPTCache)** שנוצרה כחלק מפרויקט אקדמי. המטרה המרכזית של פרויקט זה היא להציג וליישם מדיניות פינוי (Eviction Policy) משופרת עבור זיכרון מטמון (Cache) בסביבות של מודלי שפה גדולים (LLMs).

**הבעיה:** מדיניות פינוי מסורתית כמו **Least Recently Used (LRU)** אינה יעילה מספיק עבור LLMs. היא מתעלמת מעלות היצירה של כל פריט ב-cache ועלולה לזרוק פריטים יקרים חישובית לטובת פריטים זולים, רק בגלל שהגישה אליהם התבצעה לאחרונה.

**הפתרון שלנו:** פיתחנו מדיניות פינוי חדשה בשם `CostAwareCacheEviction`. מדיניות זו מחשבת "ערך" דינמי לכל פריט ב-cache על בסיס שלושה פרמטרים מרכזיים:
1.  **עלות יצירה (Base Cost):** כמה זמן ויקר היה לייצר את התשובה במקור.
2.  **תדירות גישה (Popularity):** באיזו תדירות משתמשים בפריט זה.
3.  **גיל (Age):** כמה זמן הפריט נמצא ב-cache, עם דעיכה בערך לאורך זמן.

בדרך זו, אנו מבטיחים שהפריטים היקרים והשימושיים ביותר יישארו ב-cache, מה שמוביל לשיפור משמעותי בביצועים.

## 🚀 תכונות מרכזיות של ההרחבה

* **מדיניות פינוי חכמה (`CostAwareCacheEviction`):** מחליפה את LRU הסטנדרטי במנגנון שממקסם את הערך הכלכלי של ה-cache.
* **אינטגרציה מלאה:** המדיניות החדשה נבנתה כירושה ממחלקות הבסיס של GPTCache, מה שמבטיח תאימות מלאה ושימוש קל.
* **מודולריות וגמישות:** ניתן להתאים את פונקציית חישוב העלות לצרכים ספציפיים.
* **מערך בדיקות (Benchmark) מקיף:** פיתחנו סוויטת בדיקות שמאפשרת להריץ ולהשוות בקלות בין מדיניות פינוי שונות תחת עומסי עבודה מגוונים.

## 😊 שימוש מהיר במדיניות החדשה

כדי להשתמש במדיניות הפינוי החדשה, יש לציין `eviction="CostAware"` בעת אתחול ה-`DataManager`.

```python
from gptcache import cache
from gptcache.embedding import Onnx
from gptcache.manager import get_data_manager
from gptcache.adapter import openai

# הגדרות כלליות
onnx = Onnx()

# אתחול ה-DataManager עם המדיניות החדשה
# שימו לב לפרמטר eviction="CostAware"
data_manager = get_data_manager(
    scalar_store="sqlite",
    vector_store="faiss",
    eviction="CostAware",
    vector_params={"dimension": onnx.dimension},
    eviction_params={"maxsize": 50}  # ניתן לשנות את גודל ה-cache
)

# אתחול כללי של GPTCache
cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager
)
cache.set_openai_key()

# מכאן והלאה, השימוש הוא רגיל לחלוטין
response = openai.ChatCompletion.create(
  model='gpt-3.5-turbo',
  messages=[
    {
        'role': 'user',
        'content': "מה זה Cost-Aware Eviction?"
    }
  ],
)