# JamesBot
End-to-end trainable goal-oriented conversational agent. Agent is trained to keep a distributed representation of the dialog state on multiple temporal scales. This gives the agent ability to have contextual conversation as well as optimize long term goals.

### Sample conversation
Agent trained on [Frames](https://datasets.maluuba.com/Frames) dataset. Agent is pre-trained using maximum likelihood and then actor-critic algorithm is employed to maximize BLEU reward.
```
> Hey! I'm looking for a flight to toronto.
Agent (speak, 20.74): from which city would you like to depart from ? <EOS>

> I'm from calgary.
Agent (speak, 42.89): and when would you like to travel ? <EOS>

> August.
Query: {'dst_city': 'toronto', 'or_city': 'calgary', 'str_date': 'august'}
NO RESULT
Agent (speak, 41.67): there are no results from calgary to toronto . <EOS>

> What If I were to go from vancouver on august 26?
Query: {'dst_city': 'toronto', 'or_city': 'vancouver', 'str_date': 'august 26'}
NO RESULT
Agent (speak, 34.32): there 's nothing available from vancouver to toronto . <EOS>

> Okay. Thans.
Agent (end, 9.99): my apologies . have a nice day ! <EOS>
```
