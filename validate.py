def exercise_1_a(answer):

    import sympy

    R = sympy.Symbol('R', positive=True) # Distance from the terminal position
    c = sympy.Symbol('c', positive=True) # speed of light
    v = sympy.Symbol('v', positive=True) # Particle velocity
    q = sympy.Symbol('q', positive=True) # Electric charge
    a = sympy.Symbol('a', positive=True) # Acceleration

    correct = R/c

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_1_b(answer):

    import sympy

    R = sympy.Symbol('R', positive=True) # Distance from the terminal position
    c = sympy.Symbol('c', positive=True) # speed of light
    v = sympy.Symbol('v', positive=True) # Particle velocity
    q = sympy.Symbol('q', positive=True) # Electric charge
    a = sympy.Symbol('a', positive=True) # Acceleration

    correct = R*v/c

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_1_c(answer):

    import sympy

    R = sympy.Symbol('R', positive=True) # Distance from the terminal position
    c = sympy.Symbol('c', positive=True) # speed of light
    v = sympy.Symbol('v', positive=True) # Particle velocity
    q = sympy.Symbol('q', positive=True) # Electric charge
    a = sympy.Symbol('a', positive=True) # Acceleration

    correct = v/a

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_1_d(answer):

    import sympy

    R = sympy.Symbol('R', positive=True) # Distance from the terminal position
    c = sympy.Symbol('c', positive=True) # speed of light
    v = sympy.Symbol('v', positive=True) # Particle velocity
    q = sympy.Symbol('q', positive=True) # Electric charge
    a = sympy.Symbol('a', positive=True) # Acceleration

    correct = q/R**2

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_1_e(answer):

    import sympy

    R = sympy.Symbol('R', positive=True) # Distance from the terminal position
    c = sympy.Symbol('c', positive=True) # speed of light
    v = sympy.Symbol('v', positive=True) # Particle velocity
    q = sympy.Symbol('q', positive=True) # Electric charge
    a = sympy.Symbol('a', positive=True) # Acceleration

    correct = a*q/R/c**2

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_1_f(answer):

    import sympy

    R = sympy.Symbol('R', positive=True) # Distance from the terminal position
    c = sympy.Symbol('c', positive=True) # speed of light
    v = sympy.Symbol('v', positive=True) # Particle velocity
    q = sympy.Symbol('q', positive=True) # Electric charge
    a = sympy.Symbol('a', positive=True) # Acceleration

    correct = a**2*q**2/R**2/c**4

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_1_g(answer):

    import sympy

    R = sympy.Symbol('R', positive=True) # Distance from the terminal position
    c = sympy.Symbol('c', positive=True) # speed of light
    v = sympy.Symbol('v', positive=True) # Particle velocity
    q = sympy.Symbol('q', positive=True) # Electric charge
    a = sympy.Symbol('a', positive=True) # Acceleration

    correct = a**2*q**2/R**2/c**3

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_1_h(answer):

    import sympy

    R = sympy.Symbol('R', positive=True) # Distance from the terminal position
    c = sympy.Symbol('c', positive=True) # speed of light
    v = sympy.Symbol('v', positive=True) # Particle velocity
    q = sympy.Symbol('q', positive=True) # Electric charge
    a = sympy.Symbol('a', positive=True) # Acceleration

    correct = a**2*q**2/c**3

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_2_a(answer):

    import sympy

    h = sympy.Symbol('h', positive=True) # Planck constant
    v = sympy.Symbol('v', positive=True) # Velocity
    m = sympy.Symbol('m', positive=True) # Electron mass
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light
    n = sympy.Symbol('n', positive=True) # Number density

    correct = h/m/v

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_2_b(answer):

    import sympy

    h = sympy.Symbol('h', positive=True) # Planck constant
    v = sympy.Symbol('v', positive=True) # Velocity
    m = sympy.Symbol('m', positive=True) # Electron mass
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light
    n = sympy.Symbol('n', positive=True) # Number density

    correct = m*q**2*v**2/h**2

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_2_c(answer):

    import sympy

    h = sympy.Symbol('h', positive=True) # Planck constant
    v = sympy.Symbol('v', positive=True) # Velocity
    m = sympy.Symbol('m', positive=True) # Electron mass
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light
    n = sympy.Symbol('n', positive=True) # Number density

    correct = m**2*q**6*v**4/c**3/h**4

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_2_d(answer):

    import sympy

    h = sympy.Symbol('h', positive=True) # Planck constant
    v = sympy.Symbol('v', positive=True) # Velocity
    m = sympy.Symbol('m', positive=True) # Electron mass
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light
    n = sympy.Symbol('n', positive=True) # Number density

    correct = h/m/v**2

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_2_e(answer):

    import sympy

    h = sympy.Symbol('h', positive=True) # Planck constant
    v = sympy.Symbol('v', positive=True) # Velocity
    m = sympy.Symbol('m', positive=True) # Electron mass
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light
    n = sympy.Symbol('n', positive=True) # Number density

    correct = m*q**6*v**2/c**3/h**3

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_2_f(answer):

    import sympy

    h = sympy.Symbol('h', positive=True) # Planck constant
    v = sympy.Symbol('v', positive=True) # Velocity
    m = sympy.Symbol('m', positive=True) # Electron mass
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light
    n = sympy.Symbol('n', positive=True) # Number density

    correct = h**2/m**2/v**2

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_2_g(answer):

    import sympy

    h = sympy.Symbol('h', positive=True) # Planck constant
    v = sympy.Symbol('v', positive=True) # Velocity
    m = sympy.Symbol('m', positive=True) # Electron mass
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light
    n = sympy.Symbol('n', positive=True) # Number density

    correct = m**2*v/h**2/n

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_2_h(answer):

    import sympy

    h = sympy.Symbol('h', positive=True) # Planck constant
    v = sympy.Symbol('v', positive=True) # Velocity
    m = sympy.Symbol('m', positive=True) # Electron mass
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light
    n = sympy.Symbol('n', positive=True) # Number density

    correct = n*q**6*v/c**3/h/m

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_2_i(answer):

    import sympy

    h = sympy.Symbol('h', positive=True) # Planck constant
    v = sympy.Symbol('v', positive=True) # Velocity
    m = sympy.Symbol('m', positive=True) # Electron mass
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light
    n = sympy.Symbol('n', positive=True) # Number density

    correct = n**2*q**6*v/c**3/h/m

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_3_a(answer):

    import sympy

    q = sympy.Symbol('q', positive=True) # Elementary charge
    E = sympy.Symbol('E', positive=True) # Amplitude of the EM field
    m = sympy.Symbol('m', positive=True) # Mass of the electron
    c = sympy.Symbol('c', positive=True) # Speed of light

    correct = E*q/m

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_3_b(answer):

    import sympy

    q = sympy.Symbol('q', positive=True) # Elementary charge
    E = sympy.Symbol('E', positive=True) # Amplitude of the EM field
    m = sympy.Symbol('m', positive=True) # Mass of the electron
    c = sympy.Symbol('c', positive=True) # Speed of light

    correct = E**2*q**4/c**3/m**2

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_3_c(answer):

    import sympy

    q = sympy.Symbol('q', positive=True) # Elementary charge
    E = sympy.Symbol('E', positive=True) # Amplitude of the EM field
    m = sympy.Symbol('m', positive=True) # Mass of the electron
    c = sympy.Symbol('c', positive=True) # Speed of light

    correct = E**2*c

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_3_d(answer):

    import sympy

    q = sympy.Symbol('q', positive=True) # Elementary charge
    E = sympy.Symbol('E', positive=True) # Amplitude of the EM field
    m = sympy.Symbol('m', positive=True) # Mass of the electron
    c = sympy.Symbol('c', positive=True) # Speed of light

    correct = q**4/c**4/m**2

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_4_a(answer):

    import sympy

    nu = sympy.Symbol('nu', positive=True) # Frequency
    c = sympy.Symbol('c', positive=True) # Speed of light
    kT = sympy.Symbol('kT', positive=True) # Temperature

    correct = c/nu

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_4_b(answer):

    import sympy

    nu = sympy.Symbol('nu', positive=True) # Frequency
    c = sympy.Symbol('c', positive=True) # Speed of light
    kT = sympy.Symbol('kT', positive=True) # Temperature

    correct = kT*nu**3/c**3
    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_4_c(answer):

    import sympy

    nu = sympy.Symbol('nu', positive=True) # Frequency
    c = sympy.Symbol('c', positive=True) # Speed of light
    kT = sympy.Symbol('kT', positive=True) # Temperature

    correct = kT*nu**2/c**2

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_5_a(answer):

    import sympy

    n = sympy.Symbol('n', positive=True) # Number density
    q = sympy.Symbol('q', positive=True) # Elementary charge
    m = sympy.Symbol('m', positive=True) # Electron mass
    kT = sympy.Symbol('kT', positive=True) # Temperature
    c = sympy.Symbol('c', positive=True) # Speed of light
    nu = sympy.Symbol('nu', positive=True) # Frequency
    h = sympy.Symbol('h', positive=True) # Planck constant

    correct = n**2*q**6/(kT*m)**sympy.Rational(3,2)/c/nu**2

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_5_b(answer):

    import sympy

    n = sympy.Symbol('n', positive=True) # Number density
    q = sympy.Symbol('q', positive=True) # Elementary charge
    m = sympy.Symbol('m', positive=True) # Electron mass
    kT = sympy.Symbol('kT', positive=True) # Temperature
    c = sympy.Symbol('c', positive=True) # Speed of light
    nu = sympy.Symbol('nu', positive=True) # Frequency
    h = sympy.Symbol('h', positive=True) # Planck constant

    correct = h**2*n**2*q**6/(kT)**sympy.Rational(7,2)/c/m**sympy.Rational(3,2)

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_5_c(answer):

    import sympy

    n = sympy.Symbol('n', positive=True) # Number density
    q = sympy.Symbol('q', positive=True) # Elementary charge
    m = sympy.Symbol('m', positive=True) # Electron mass
    kT = sympy.Symbol('kT', positive=True) # Temperature
    c = sympy.Symbol('c', positive=True) # Speed of light
    nu = sympy.Symbol('nu', positive=True) # Frequency
    h = sympy.Symbol('h', positive=True) # Planck constant

    correct = h**2*n*q**6/(kT)**sympy.Rational(7,2)/c/m**sympy.Rational(3,2)

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_5_d(answer):

    import sympy

    n = sympy.Symbol('n', positive=True) # Number density
    q = sympy.Symbol('q', positive=True) # Elementary charge
    m = sympy.Symbol('m', positive=True) # Electron mass
    T = sympy.Symbol('T', positive=True) # Temperature
    c = sympy.Symbol('c', positive=True) # Speed of light
    nu = sympy.Symbol('nu', positive=True) # Frequency
    h = sympy.Symbol('h', positive=True) # Planck constant

    correct = (c**sympy.Rational(6,7)*
               h**sympy.Rational(4,7)*
               m**sympy.Rational(1,7)*
               n**sympy.Rational(2,7)*
               q**sympy.Rational(4,7))

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_6_a(answer):

    import sympy

    kT_c = sympy.Symbol('kT_c', positive=True) # Core temperature
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    M = sympy.Symbol('M', positive=True) # Mass of the star
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass
    m_e = sympy.Symbol('m_e', positive=True) # Electron mass
    h = sympy.Symbol('h', positive=True) # Planck constant
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light

    correct = G*M*m_p/kT_c

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_6_b(answer):

    import sympy

    kT_c = sympy.Symbol('kT_c', positive=True) # Core temperature
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    M = sympy.Symbol('M', positive=True) # Mass of the star
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass
    m_e = sympy.Symbol('m_e', positive=True) # Electron mass
    h = sympy.Symbol('h', positive=True) # Planck constant
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light

    correct = sympy.sqrt(kT_c)*h**2*q**6/G**4/M**2/c**2/m_p**6/m_e**sympy.Rational(3,2)

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_6_c(answer):

    import sympy

    kT_c = sympy.Symbol('kT_c', positive=True) # Core temperature
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    M = sympy.Symbol('M', positive=True) # Mass of the star
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass
    m_e = sympy.Symbol('m_e', positive=True) # Electron mass
    h = sympy.Symbol('h', positive=True) # Planck constant
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light

    correct = G**7*M**5*sympy.sqrt(kT_c)*m_e**sympy.Rational(3,2)*m_p**9/c/h**5/q**6

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_6_d(answer):

    import sympy

    kT_c = sympy.Symbol('kT_c', positive=True) # Core temperature
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    M = sympy.Symbol('M', positive=True) # Mass of the star
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass
    m_e = sympy.Symbol('m_e', positive=True) # Electron mass
    h = sympy.Symbol('h', positive=True) # Planck constant
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light

    correct = kT_c*q**4/G/c**5/m_e**2/m_p**2

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_6_e(answer):

    import sympy

    kT_c = sympy.Symbol('T_c', positive=True) # Core temperature
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    M = sympy.Symbol('M', positive=True) # Mass of the star
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass
    m_e = sympy.Symbol('m_e', positive=True) # Electron mass
    h = sympy.Symbol('h', positive=True) # Planck constant
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light

    correct = G**4*M**3*c**2*m_e**2*m_p**5/h**3/q**4

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_6_f(answer):

    import sympy

    kT_c = sympy.Symbol('kT_c', positive=True) # Core temperature
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    M = sympy.Symbol('M', positive=True) # Mass of the star
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass
    m_e = sympy.Symbol('m_e', positive=True) # Electron mass
    h = sympy.Symbol('h', positive=True) # Planck constant
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light

    correct = (G*M**2*c**3*h**3)**sympy.Rational(1,4)/kT_c

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_6_g(answer):

    import sympy

    kT_c = sympy.Symbol('kT_c', positive=True) # Core temperature
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    M = sympy.Symbol('M', positive=True) # Mass of the star
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass
    m_e = sympy.Symbol('m_e', positive=True) # Electron mass
    h = sympy.Symbol('h', positive=True) # Planck constant
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light

    correct = (sympy.sqrt(M)*
               kT_c*
               q**4/
               G**sympy.Rational(1,4)/
               c**sympy.Rational(23,4)/
               h**sympy.Rational(3,4)/
               m_e**2/
               m_p)

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_6_h(answer):

    import sympy

    kT_c = sympy.Symbol('kT_c', positive=True) # Core temperature
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    M = sympy.Symbol('M', positive=True) # Mass of the star
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass
    m_e = sympy.Symbol('m_e', positive=True) # Electron mass
    h = sympy.Symbol('h', positive=True) # Planck constant
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light

    correct = G*M*c**5*m_e**2*m_p/q**4

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_6_i(answer):

    import sympy

    kT_c = sympy.Symbol('kT_c', positive=True) # Core temperature
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    M = sympy.Symbol('M', positive=True) # Mass of the star
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass
    m_e = sympy.Symbol('m_e', positive=True) # Electron mass
    h = sympy.Symbol('h', positive=True) # Planck constant
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light

    correct = (c**sympy.Rational(3,2)*
               h*m_e**sympy.Rational(1,4)*q/
               G**sympy.Rational(3,2)/
               (kT_c)**sympy.Rational(1,4)/
               m_p**2)

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_6_j(answer):

    import sympy

    T_c = sympy.Symbol('T_c', positive=True) # Core temperature
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    M = sympy.Symbol('M', positive=True) # Mass of the star
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass
    m_e = sympy.Symbol('m_e', positive=True) # Electron mass
    h = sympy.Symbol('h', positive=True) # Planck constant
    q = sympy.Symbol('q', positive=True) # Elementary charge
    c = sympy.Symbol('c', positive=True) # Speed of light

    correct = sympy.sqrt(c**3*h**3/G**3/m_p**4)

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_7_a(answer):

    import sympy

    n = sympy.Symbol('n', positive=True) # Number density
    h = sympy.Symbol('h', positive=True) # Planck constant
    m_e = sympy.Symbol('m_e', positive=True) # Electron mass
    M = sympy.Symbol('M', positive=True) # Object mass
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    kT = sympy.Symbol('kT', positive=True) # Necessary temperature for nuclear burning

    correct = n**sympy.Rational(-1,3)

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_7_b(answer):

    import sympy

    n = sympy.Symbol('n', positive=True) # Number density
    h = sympy.Symbol('h', positive=True) # Planck constant
    m_e = sympy.Symbol('m_e', positive=True) # Electron mass
    M = sympy.Symbol('M', positive=True) # Object mass
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    kT = sympy.Symbol('kT', positive=True) # Necessary temperature for nuclear burning

    correct = h*n**sympy.Rational(1,3)

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_7_c(answer):

    import sympy

    n = sympy.Symbol('n', positive=True) # Number density
    h = sympy.Symbol('h', positive=True) # Planck constant
    m_e = sympy.Symbol('m_e', positive=True) # Electron mass
    M = sympy.Symbol('M', positive=True) # Object mass
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    kT = sympy.Symbol('kT', positive=True) # Necessary temperature for nuclear burning

    correct = h**2*n**sympy.Rational(2,3)/m_e

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_7_d(answer):

    import sympy

    n = sympy.Symbol('n', positive=True) # Number density
    h = sympy.Symbol('h', positive=True) # Planck constant
    m_e = sympy.Symbol('m_e', positive=True) # Electron mass
    M = sympy.Symbol('M', positive=True) # Object mass
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    kT = sympy.Symbol('kT', positive=True) # Necessary temperature for nuclear burning

    correct = h**2*n**sympy.Rational(5,3)/m_e

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_7_e(answer):

    import sympy

    n = sympy.Symbol('n', positive=True) # Number density
    h = sympy.Symbol('h', positive=True) # Planck constant
    m_e = sympy.Symbol('m_e', positive=True) # Electron mass
    M = sympy.Symbol('M', positive=True) # Object mass
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    kT = sympy.Symbol('kT', positive=True) # Necessary temperature for nuclear burning

    correct = h**2/G/M**sympy.Rational(1,3)/m_e/m_p**sympy.Rational(5,3)

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_7_f(answer):

    import sympy

    n = sympy.Symbol('n', positive=True) # Number density
    h = sympy.Symbol('h', positive=True) # Planck constant
    m_e = sympy.Symbol('m_e', positive=True) # Electron mass
    M = sympy.Symbol('M', positive=True) # Object mass
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    kT = sympy.Symbol('kT', positive=True) # Necessary temperature for nuclear burning

    correct = (kT**3*h**6/G**6/m_e**3/m_p**8)**sympy.Rational(1,4)

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_8_a(answer):

    import sympy

    G = sympy.Symbol('G', positive=True) # Gravitation constant
    h = sympy.Symbol('h', positive=True) # Planck constant
    kT = sympy.Symbol('kT', positive=True) # Nuclear burning temperature
    c = sympy.Symbol('c', positive=True) # Speed of light
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass

    correct = sympy.sqrt(c**7*h**3/G**3/kT**2/m_p**2)

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_9_a(answer):

    import sympy

    n = sympy.Symbol('n', positive=True) # Number density
    h = sympy.Symbol('h', positive=True) # Planck constant
    c = sympy.Symbol('c', positive=True) # Speed of light
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass

    correct = h*n**sympy.Rational(1,3)

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_9_b(answer):

    import sympy

    n = sympy.Symbol('n', positive=True) # Number density
    h = sympy.Symbol('h', positive=True) # Planck constant
    c = sympy.Symbol('c', positive=True) # Speed of light
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass

    correct = c*h*n**sympy.Rational(1,3)

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_9_c(answer):

    import sympy

    n = sympy.Symbol('n', positive=True) # Number density
    h = sympy.Symbol('h', positive=True) # Planck constant
    c = sympy.Symbol('c', positive=True) # Speed of light
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass

    correct = c*h*n**sympy.Rational(4,3)

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_9_d(answer):

    import sympy

    n = sympy.Symbol('n', positive=True) # Number density
    h = sympy.Symbol('h', positive=True) # Planck constant
    c = sympy.Symbol('c', positive=True) # Speed of light
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    m_p = sympy.Symbol('m_p', positive=True) # Proton mass

    correct = sympy.sqrt(c**3*h**3/G**3/m_p**4)

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_10_a(answer):

    import sympy

    m_pi = sympy.Symbol(r'm_{\pi}', positive=True) # Pion mass
    h = sympy.Symbol('h', positive=True) # Planck constant
    c = sympy.Symbol('c', positive=True) # Speed of light
    m_n = sympy.Symbol('m_n', positive=True) # Nucleon mass

    correct = h/m_pi/c

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_10_b(answer):

    import sympy

    m_pi = sympy.Symbol(r'm_{\pi}', positive=True) # Pion mass
    h = sympy.Symbol('h', positive=True) # Planck constant
    c = sympy.Symbol('c', positive=True) # Speed of light
    m_n = sympy.Symbol('m_n', positive=True) # Nucleon mass

    correct = c**3*m_n*m_pi**3/h**3

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_10_c(answer):

    import sympy

    m_pi = sympy.Symbol(r'm_{\pi}', positive=True) # Pion mass
    h = sympy.Symbol('h', positive=True) # Planck constant
    c = sympy.Symbol('c', positive=True) # Speed of light
    m_n = sympy.Symbol('m_n', positive=True) # Nucleon mass
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    M = sympy.Symbol('M', positive=True) # Mass

    correct = G*M/c**2

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'

def exercise_10_d(answer):

    import sympy

    m_pi = sympy.Symbol(r'm_{\pi}', positive=True) # Pion mass
    h = sympy.Symbol('h', positive=True) # Planck constant
    c = sympy.Symbol('c', positive=True) # Speed of light
    m_n = sympy.Symbol('m_n', positive=True) # Nucleon mass
    G = sympy.Symbol('G', positive=True) # Gravitation constant
    M = sympy.Symbol('M', positive=True) # Mass

    correct = sympy.sqrt(c**3*h**3/G**3/m_n/m_pi**3)

    if answer == correct:
        return 'Correct'
    temp = answer/correct
    temp = temp.simplify().n()
    if type(temp)==type(sympy.Rational(1,2).n()):
        return "Correct, but we won't need the numerical prefactors here"
    return 'Try again'
