# Informaticup
Der InformatiCup ist ein Wettbewerb, der von der Gesellschaft für Informatik jährlich veranstaltet wird. Studierende aller Fachrichtungen an Universitäten und Hochschulen in Deutschland, Österreich und der Schweiz dürfen daran teilnehmen.

Für den Informaticup 2023 sollte ein Programm entwickelt werden, das das Spiel "Profit!" auf maximale Punktzahl bei gleichzeitig geringer Rundenzahl optimiert. 

Dieses Repository enthält die Lösung des Teams “Die Schmetterlinge!” bestehend aus Lisa Binkert, Leopold Gaube und Yasin Koschinski. Wir sind gespannt auf die Lösungen der anderen Teams.


# Benutzerhandbuch

## Docker
Um unser Programm über einen Docker-Container laufen zu lassen, müssen folgende Kommandos ausgeführt werden:  
`docker build . --tag <tag_name>`  
`docker run -i --rm --network none --cpus 2.000 --memory 2G --memory-swap 2g <tag_name>`   

Im Anschluss erwartet die Standardeingabe eine Aufgabe im json-Format. Das Programm errechnet eine Lösung und gibt diese über die Standardausgabe als Liste von platzierbaren Objekten zurück. Dies kann bei Aufgaben mit einem größen Spielfeld mehrere Minuten dauern.

Alternativ lässt sich eine Eingabe auch direkt an den Container geben, indem man Pipes nutzt (die JSON-Aufgabe muss dabei unter Linux in Hochkommas gesetzt werden):  
`echo '{"width":40,"height":20,"objects": ... }' | docker run -i --rm --network none --cpus 2.000 --memory 2G --memory-swap 2g <tag_name>`

## Einstellungen
Um unsere Lösung anzupassen, können einfach die Einstellungen verändert werden.

Alle Hyperparameter der Agenten, Einstellungen für den Task Manager, sowie weitere Debug Informationen lassen sich gesammelt in der `settings.py` Datei einstellen. 
Es kann beispielsweise eine andere Modell-Architektur für den untergeordneten Agenten gewählt oder die Hinderniswahrscheinlichkeit des Task Generators angepasst werden. Auch das zuvor erwähnte vereinfachte Spiel kann durch Setzen von `SIMPLE_GAME = True` aktiviert werden. 

## Trainieren eines untergeordneten Agenten
Im Anschluss an Änderungen in den Einstellungen muss ein neuer untergeordneter Agent mittels `train_model.py` trainiert werden. Dies kann je nach Einstellungen und verwendeter Hardware mehrere Stunden in Anspruch nehmen. Um die Trainingszeit zu verringern, sollte in Betracht gezogen werden, die maximale Episodenzahl in `settings.py` zu reduzieren.

## Evaluation von untergeordneten Agenten
Nachdem mehrere Agenten trainiert wurden, kann mithilfe von `evaluate_model.py` überprüft werden, welches Modell die meisten Aufgaben des Task Generators lösen kann und sich somit am besten in einem Hindernis-Labyrinth bewegt. Standardmäßig wird auf verschiedenen Spielfeldgrößen von 20x20, 30x30 und 50x50 evaluiert.
Das Ergebnis dient nur als relativer Anhaltspunkt. Es sagt nichts darüber aus, wie gut der Agent mit “echten” Aufgaben zurechtkommt, da die Aufgaben des Task Generators sich von Menschen erstellten Aufgaben unterscheiden. 

## Lösen von Aufgaben
Wird `solve_game.py` mit einem Pfad zu einer json-Datei aufgerufen, wird nur diese Aufgabe gelöst. Alternativ kann mit “solve” als Parameter auch eine Aufgabe über die Standardeingabe eingelesen werden. Es werden alle Ausgaben auf die Standardausgabe unterdrückt, bis zum Schluss eine Lösung als Liste von platzierbaren Gebäuden zurückgegeben wird. 

Das Ausführen von `solve_game.py` ohne weitere Parameter bewirkt, dass automatisch alle von uns definierten Aufgaben nacheinander gelöst werden. Hierbei werden auch die initiale Lösung sowie jede Verbesserung davon auf der Standardausgabe ausgegeben. Um sich einen ausführlichen Lösungsweg anzeigen zu lassen, sollte in settings.py DEBUG=True gesetzt werden. Damit werden auch fehlgeschlagene Verbindungswege schrittweise angezeigt.

## Unittests
Durch Aufruf von `all_unit_test.py` werden alle in 4.3.4 angegebenen Unittests durchlaufen und auf mögliche Fehler hingewiesen.

