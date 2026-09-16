(* ::Package:: *)

(* analytic_wait_model.wl *)
BeginPackage["WarehouseOptimization`"]

GetAnalyticJ::usage = "GetAnalyticJ[L, w, v, deltaExp] berechnet die symbolische Kostenfunktion J in Abh\[ADoubleDot]ngigkeit von Global`wait."

Begin["`Private`"]

GetAnalyticJ[Lval_, wval_, vval_, deltaVal_] := Module[
    {L = Lval, w = wval, v = vval, deltaExp = deltaVal, lambda, dist, troute, 
     EDges, routeshare, candidateshare, ArrivalTime3Orders, ECostAfter},

    routeshare = 3/8;
    candidateshare = 5/8;
    lambda = 1/deltaExp;
    dist = ExponentialDistribution[lambda];
    
    (* troute ist die interne Zeit der Route: t = tabs - wait *)
    troute[tabs_] := tabs - Global`wait;

    (* Wir h\[UDoubleDot]llen die gesamte Berechnung in Assuming ein, 
       damit Integrate die Grenzen (wait + X) als reell akzeptiert *)
    Assuming[Global`wait > 0 && Element[Global`wait, Reals],
        EDges = (
            (* I0 & I1: Integrale \[UDoubleDot]ber 0 sind 0 *)
            0 + 
            
            (* I2: Vertikaler Weg Gasse 8 *)
            Integrate[
                (routeshare * ((troute[tabs] - 8)/(3*L)) * (troute[tabs] - 8)) * PDF[dist, tabs], 
                {tabs, Global`wait + 8, Global`wait + 25}
            ] +

            (* I3: Horizontal Gasse 8 zu 2 *)
            Integrate[
                ( (routeshare * (1/3) * (2*troute[tabs] - 33)) + 
                  ((candidateshare - (6/8)*((33 - troute[tabs]) - 2)/6) * (troute[tabs] - 25))
                ) * PDF[dist, tabs], 
                {tabs, Global`wait + 25, Global`wait + 31}
            ] +

            (* I4: Vertikal Gasse 2 *)
            Integrate[
                ( (routeshare * ((troute[tabs] - 14)/(3*L)) * (( (troute[tabs]-31)/(troute[tabs]-31+L)*(troute[tabs]-31) ) + 
                     (L/(troute[tabs]-31+L)*(L + 2*(6 + (troute[tabs]-31)))))
                  ) + (candidateshare * 6)
                ) * PDF[dist, tabs], 
                {tabs, Global`wait + 31, Global`wait + 48}
            ] +

            (* I5: Horizontal Gasse 2 zu 1 *)
            Integrate[
                ( (routeshare * (2/3) * (2*troute[tabs] - 56)) + 
                  (candidateshare * 2 * (4 - ((50 - troute[tabs]) - 1)))
                ) * PDF[dist, tabs], 
                {tabs, Global`wait + 48, Global`wait + 49}
            ] +

            (* I6a: Vertikal Return Gasse 1 (oben) *)
            Integrate[
                (routeshare * (2/3) * 42 + candidateshare * 8) * PDF[dist, tabs], 
                {tabs, Global`wait + 49, Global`wait + 66}
            ] +

            (* I6b: Vertikal Return Gasse 1 (unten) *)
            Integrate[
                ( (routeshare * ((troute[tabs] - 32)/(3*L)) * (( (troute[tabs]-66)/(troute[tabs]-66+2*L)*(troute[tabs]-66) ) + 
                     (L/(troute[tabs]-66+2*L)*(L + 2)) + 
                     (L/(troute[tabs]-66+2*L)*65))
                  ) + (candidateshare * 42)
                ) * PDF[dist, tabs], 
                {tabs, Global`wait + 66, Global`wait + 83}
            ] +

            (* I7: Horizontal Gasse 1 zu Depot *)
            Integrate[
                ( (routeshare * (2*troute[tabs] - 124)) + 
                  (candidateshare * (2*(5 - (84 - troute[tabs])) + 2*L))
                ) * PDF[dist, tabs], 
                {tabs, Global`wait + 83, Global`wait + 84}
            ] +

            (* I8: Nach R\[UDoubleDot]ckkehr *)
            (
                ArrivalTime3Orders = 3/lambda;
                ECostAfter = ArrivalTime3Orders + 84;
                Integrate[ECostAfter * PDF[dist, tabs], {tabs, Global`wait + 84, Infinity}]
            )
        );

        (* Die Formel massiv vereinfachen, bevor sie an Python geht *)
        FullSimplify[EDges + Global`wait]
    ]
]

End[]
EndPackage[]
