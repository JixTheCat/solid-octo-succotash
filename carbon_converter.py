

def scope2ef(state="national"):
    """return the scope 2 emission factor for kWh conversion.
    
    for reference: https://www.dcceew.gov.au/sites/default/files/documents/national-greenhouse-accounts-factors-2022.pdf
    
    Args:
        state: [str] The abbreviation for the state i.e SA, NSW etc.
            The defualt is national, for the national average. Note that
            WA is split into SWIS and NWIS subsections, it defaults to south.

    Returns:
        Emission factor in the form of a float.
    """
    match state.lower():
        case "nsw":
            return 0.73
        case "act":
            return 0.73
        case "vic":
            return 0.85
        case "qld":
            return 0.73
        case "sa":
            return  0.25
        case "tas":
            return 0.17
        case "swis":
            return 0.51
        case "wa":
            return 0.51
        case "nt":
            return 0.54
        case "national":
            return 0.68            


def scope2(Q: float, EF=scope2ef()):
    """Converts kW/H into CO2 emissions given an emissions factor
    
    Args:
        Q: [float] of kW/Hs
        EF: [float] of emission factor, obtainable from the scope2ef function.
        
    Returns: [float] emissions in CO2 equivalent
    """
    return Q * (EF/1000)


def convert(Q: float, EC: float, EF1: float, EF3: float):
    """The combustion gas emeission forumla from DCCCEEW

    Args:
        Q: is the quantity of fuel type measured in kilolitres,
            tonnes or gigajoules.
        EC: is the energy content factor of the fuel (gigajoules per
            tonne) according to each fuel in Table 6
        EF1: is the scope 1 emission factor, in kilograms of CO2-e
            per gigajoule, for each gas type and for each
        fuel type as per Table 6. These can also be added together to
            get a combined emission factor, to ensure CO2, CH4 and
            N2O emissions are included, as per Example 5.
        EF3: is the scope 3 emission factor, in kilograms of CO2-e
            per gigajoule, for each gas type and for each fuel type
            as per Table 6.

    Returns:
        t CO2-e is the emissions of each gas type from each fuel type
        measured in CO2-e tonnes.
    """
    return (Q * EC * (EF1 + EF3)) / 1000


def diesel_irrigation(Q: float):
    """converts tonnes diesel used in irrigation to CO2/tonnes

    Args:
        Q: The quantity of fuel type measured in Megalitres.

    Returns:
        t CO2-e is the emissions of each gas type from each fuel type
        measured in CO2-e tonnes.
    """
    return convert(Q=Q*1000, EC=38.6, EF1=70.2, EF3=17.3)


def diesel_vehicle(Q: float):
    """converts tonnes diesel used in vehicles to CO2/tonnes.

    This uses the average between different diesel vehicle types of
    71.3625 for EF1. Noting EF3 are all identical.

    Args:
        Q: The quantity of fuel type measured in Megalitres.

    Returns:
        t CO2-e is the emissions of each gas type from each fuel type
        measured in CO2-e tonnes.
    """
    return convert(Q=Q*1000, EC=38.6, EF1=71.3625, EF3=17.3)


def petrol_irrigation(Q: float):
    """converts tonnes diesel used in irrigation to CO2/tonnes

    Args:
        Q: The quantity of fuel type measured in Megalitres.

    Returns:
        t CO2-e is the emissions of each gas type from each fuel type
        measured in CO2-e tonnes.
    """
    return convert(Q=Q*1000, EC=39.7, EF1=73.8, EF3=18)


def petrol_vehicle(Q: float):
    """converts tonnes diesel used in vehicles to CO2/tonnes.

    This uses the average between different diesel vehicle types of
    71.3625 for EF1. Noting EF3 are all identical.

    Args:
        Q: The quantity of fuel type measured in Megalitres.

    Returns:
        t CO2-e is the emissions of each gas type from each fuel type
        measured in CO2-e tonnes.
    """
    return convert(Q=Q*1000, EC=39.7, EF1=74.18, EF3=18)


def lpg_irrigation(Q: float):
    """converts tonnes diesel used in irrigation to CO2/tonnes

    Args:
        Q: The quantity of fuel type measured in Megalitres.

    Returns:
        t CO2-e is the emissions of each gas type from each fuel type
        measured in CO2-e tonnes.
    """
    return convert(Q=Q*1000, EC=25.7, EF1=60.6, EF3=20.2)


def lpg_vehicle(Q: float):
    """converts tonnes diesel used in vehicles to CO2/tonnes.

    This uses the average between different diesel vehicle types of
    71.3625 for EF1. Noting EF3 are all identical.

    Args:
        Q: The quantity of fuel type measured in Megalitres.

    Returns:
        t CO2-e is the emissions of each gas type from each fuel type
        measured in CO2-e tonnes.
    """
    return convert(Q=Q*1000, EC=26.2, EF1=60.6, EF3=20.2)


def biodiesel_vehicle(Q: float):
    """converts tonnes diesel used in vehicles to CO2/tonnes.

    This uses the average between different diesel vehicle types of
    71.3625 for EF1. Noting EF3 are all identical.

    Args:
        Q: The quantity of fuel type measured in Megalitres.

    Returns:
        t CO2-e is the emissions of each gas type from each fuel type
        measured in CO2-e tonnes.
    """
    return convert(Q=Q*1000, EC=34.6, EF1=2.5, EF3=0)


def biodiesel_irrigation(Q: float):
    """converts tonnes diesel used in irrigation to CO2/tonnes

    Args:
        Q: The quantity of fuel type measured in Megalitres.

    Returns:
        t CO2-e is the emissions of each gas type from each fuel type
        measured in CO2-e tonnes.
    """
    return convert(Q=Q*1000, EC=34.6, EF1=.28, EF3=0)
