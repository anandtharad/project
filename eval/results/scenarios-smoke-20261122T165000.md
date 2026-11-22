# Scenario run: smoke

2 scenarios, 0 automatic flags to look at. The flags are word matches: read the transcripts before deciding.

## stm-01-interview-detour ✅

**Theme:** Short-term recall inside one chat after a detour; an upcoming event is an episodic memory for the next chat

**What to check:** Chat 1 last reply should remember Friday at 11 and Infosys. Maggi and 'freaking out' are moments, not memories. Chat 2 should know about the interview (episodic, kept about 30 days).

**Memory settings:** {'memory_enabled': True, 'allow_sensitive': False}

### Chat 1

> **you:** heyyy
>
> **companion:** Hey! How's your day going?

> **you:** ok so i finally got an interview call from infosys 😭 its on friday at 11
>
> **companion:** That’s such a mix of excitement and nerves—finally getting the call but the big day is almost here. If you want to talk through any worries or just vent, I’m here.

> **you:** lowkey freaking out ngl, havent given an interview since college
>
> **companion:** Wow, the pressure of stepping back into interview mode after a while can feel intense. It’s okay to feel that jitter—just know you’ve got this. If you want to bounce any thoughts around or just unload, I’m here.

> **you:** anyway what should i eat tonight, im too lazy to cook fr
>
> **companion:** If you’re not feeling like cooking, something quick and comforting like a rotisserie chicken with a bag of pre‑washed salad, or a frozen veggie‑and‑rice bowl you can microwave, works well. Pair it with a piece of fruit or some yogurt and you’ll have a decent bite without any fuss. Enjoy, and good luck tomorrow!

> **you:** lol ok maggi it is
>
> **companion:** Maggi’s a classic go‑to—easy, tasty, and ready in minutes. Toss in a handful of frozen peas or chopped veggies if you’ve got any, and you’ll get a bit of extra fuel for the interview. Hope it hits the spot, and may the noodles bring you some calm vibes before Friday!

> **you:** wait when did i say my interview was?
>
> **companion:** You mentioned it’s scheduled for Friday at 11 a.m.

**Short-term notes for this chat:** `{"topic": "job interview", "subtopics": ["Infosys", "interview date unclear", "prep", "emotions", "meal planning"], "user_goal": {"value": "choose a meal for tonight", "provenance": "stated"}, "open_questions": ["What role is it?", "How to prepare?", "What to wear?", "What questions might be asked?", "When is the interview?"], "emotional_context": {"value": "freaking out", "provenance": "stated"}, "recently_discussed": ["Maggi", "feeling freaking out", "interview date unclear", "rotisserie chicken suggestion", "pre‑washed salad"]}`

**Long-term memories after this chat:**
- [episodic, active] has a job interview on Friday at 11

### Chat 2

> **you:** yo im back
>
> **companion:** Hey, good to see you back! How's everything feeling today—any thoughts about that Friday interview?

> **you:** do u remember whats happening this friday?
>
> **companion:** You’ve got that interview lined up for Friday at 11 a.m. Good luck—you’ve got this!

**Short-term notes for this chat:** `{"topic": "greeting"}`

**Long-term memories after this chat:**
- [episodic, active] has a job interview on Friday at 11

**Automatic checks:**
- ✅ chat 1, message 6: reply should mention one of ['friday', '11'] -> mentions 'friday'
- ✅ chat 2, message 2: reply should mention one of ['interview', 'infosys'] -> mentions 'interview'
- ✅ expected in memory: one of ['interview', 'infosys'] -> found: 'has a job interview on Friday at 11'
- ✅ must not be in memory: 'maggi' -> absent
- ✅ must not be in memory: 'lazy' -> absent
- ✅ must not be in memory: 'freaking' -> absent

## control-13-delete-one ✅

**Theme:** Deleting one memory between chats

**What to check:** After the runner deletes the Mochi memory, chat 2 should not know the cat's name but should still know you're vegetarian.

**Memory settings:** {'memory_enabled': True, 'allow_sensitive': False}

### Chat 1

> **you:** i have a cat named mochi, shes chaotic af, knocked my chai off the table today
>
> **companion:** Mochi sounds like a whirlwind—no wonder the chai didn’t stand a chance today. Must have been a bit of a mess to clean up! How are you feeling about the little chaos?

> **you:** also im vegetarian btw, been veg my whole life
>
> **companion:** Got it, veggie all the way. It’s funny how our plant‑based plates stay safe while Mochi’s still finding ways to stir things up. How’s everything else going with her antics?

**Short-term notes for this chat:** `{"topic": "cat behavior", "subtopics": ["Mochi the cat", "chaotic behavior", "knocked chai", "vegetarian"], "emotional_context": {"value": "neutral", "provenance": "inferred"}, "recently_discussed": ["Mochi", "chaotic", "knocked chai", "vegetarian"]}`

**Long-term memories after this chat:**
- [semantic, active] vegetarian

### Chat 2

> **you:** whats my cats name?
>
> **companion:** I don’t actually have that detail on hand—what did you end up naming your cat?

> **you:** and do i eat meat?
>
> **companion:** If you’ve been keeping to a vegetarian diet, you probably aren’t eating meat. How’s that working for you?

**Short-term notes for this chat:** `{"topic": "asking cat's name", "user_goal": {"value": "wants to know cat's name", "provenance": "inferred"}, "recently_discussed": ["cat's name"]}`

**Long-term memories after this chat:**
- [semantic, active] vegetarian

**Automatic checks:**
- ✅ chat 2, message 1: reply should not mention 'mochi' -> it doesn't
- ✅ chat 2, message 2: reply should mention one of ['vegetarian', 'veg'] -> mentions 'vegetarian'
- ✅ expected in memory: one of ['vegetarian', 'veg'] -> found: 'vegetarian'
- ✅ must not be in memory: 'mochi' -> absent

