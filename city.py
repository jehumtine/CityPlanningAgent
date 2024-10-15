import random

class CityRule:
    def __init__(self, population_growth_rate, infrastructure_development_rate, environmental_impact_rate, environmental_conservation_rate):
        self.population_growth_rate = population_growth_rate
        self.infrastructure_development_rate = infrastructure_development_rate
        self.environmental_impact_rate = environmental_impact_rate
        self.environmental_conservation_rate = environmental_conservation_rate

    @staticmethod
    def possible_values():
        return [
            CityRule(0.1, 0.1, 0.1, 0.1),
            CityRule(0.05, 0.05, 0.01, 0.1),
            CityRule(0.01, 0.1, 0.1, 0.01),
            CityRule(0.01, 0.3, 0.1, 0.01),
            CityRule(0.1, 0.1, 0.1, 0.2),
            CityRule(0.01, 0.3, 0.1, 0.01),
            CityRule(0.1, 0.1, 0.1, 0.2),
            CityRule(0.05, 0.2, 0.15, 0.05),
            CityRule(0.02, 0.25, 0.05, 0.15),
            CityRule(0.07, 0.1, 0.2, 0.05),
            CityRule(0.03, 0.15, 0.1, 0.2),
            CityRule(0.04, 0.3, 0.1, 0.1),
            CityRule(0.06, 0.2, 0.1, 0.1),
            CityRule(0.01, 0.3, 0.1, 0.01),
            CityRule(0.1, 0.1, 0.1, 0.2),
            CityRule(0.05, 0.2, 0.15, 0.05),
            CityRule(0.02, 0.25, 0.05, 0.15),
            CityRule(0.07, 0.1, 0.2, 0.05),
            CityRule(0.03, 0.15, 0.1, 0.2),
            CityRule(0.04, 0.3, 0.1, 0.1),
            CityRule(0.06, 0.2, 0.1, 0.1),
            CityRule(0.08, 0.18, 0.12, 0.08),
            CityRule(0.09, 0.22, 0.08, 0.12),
            CityRule(0.04, 0.1, 0.2, 0.15),
            CityRule(0.03, 0.2, 0.1, 0.25),
            CityRule(0.06, 0.25, 0.15, 0.1),
            CityRule(0.02, 0.3, 0.05, 0.2),
            CityRule(0.07, 0.12, 0.18, 0.06),
            CityRule(0.1, 0.2, 0.1, 0.05),
            CityRule(0.05, 0.3, 0.1, 0.15),
            CityRule(0.08, 0.15, 0.15, 0.1),
            CityRule(0.06, 0.1, 0.2, 0.08),
            CityRule(0.04, 0.2, 0.15, 0.12)
        ]

class CellState:
    class State(str):
        EMPTY = "E"
        RESIDENTIAL = "R"
        COMMERCIAL = "C"
        INDUSTRIAL = "I"
        GREEN_SPACE = "G"

    def __init__(self, state, value):
        self.state = state
        self.value = value

    @property
    def symbol(self):
        return self.state

    @property
    def is_living(self):
        return self.state != self.State.EMPTY

    @staticmethod
    def random(config):
        rand = random.randint(0, 99)
        if config == 0:
            if rand < 50:
                return CellState(CellState.State.EMPTY, 0.1)
            elif rand < 70:
                return CellState(CellState.State.RESIDENTIAL, 0.1)
            elif rand < 85:
                return CellState(CellState.State.COMMERCIAL, 0.1)
            elif rand < 95:
                return CellState(CellState.State.INDUSTRIAL, 0.1)
            else:
                return CellState(CellState.State.GREEN_SPACE, 0.1)
        elif config == 1:
            if rand < 50:
                return CellState(CellState.State.RESIDENTIAL, 0.5)
            elif rand < 70:
                return CellState(CellState.State.EMPTY, -0.1)
            elif rand < 85:
                return CellState(CellState.State.COMMERCIAL, 0.2)
            elif rand < 95:
                return CellState(CellState.State.INDUSTRIAL, -0.2)
            else:
                return CellState(CellState.State.GREEN_SPACE, 0.3)
        elif config == 2:
            if rand < 50:
                return CellState(CellState.State.COMMERCIAL, 0.5)
            elif rand < 70:
                return CellState(CellState.State.EMPTY, -0.2)
            elif rand < 85:
                return CellState(CellState.State.RESIDENTIAL, -0.2)
            elif rand < 95:
                return CellState(CellState.State.INDUSTRIAL, 0.1)
            else:
                return CellState(CellState.State.GREEN_SPACE, -0.2)
        elif config == 3:
            if rand < 50:
                return CellState(CellState.State.INDUSTRIAL, 0.5)
            elif rand < 70:
                return CellState(CellState.State.EMPTY, -0.1)
            elif rand < 85:
                return CellState(CellState.State.RESIDENTIAL, -0.1)
            elif rand < 95:
                return CellState(CellState.State.COMMERCIAL, 0.1)
            else:
                return CellState(CellState.State.GREEN_SPACE, -0.2)
        else:
            if rand < 50:
                return CellState(CellState.State.EMPTY, 0.1)
            elif rand < 70:
                return CellState(CellState.State.RESIDENTIAL, 0.1)
            elif rand < 85:
                return CellState(CellState.State.COMMERCIAL, 0.1)
            elif rand < 95:
                return CellState(CellState.State.INDUSTRIAL, 0.1)
            else:
                return CellState(CellState.State.GREEN_SPACE, 0.1)

class CitySimulation:
    def __init__(self, size, config):
        self.size = size
        self.config = config
        self.city_state = 0.0
        self.reset_counter = 0
        self.grid = [[CellState(state=CellState.State.EMPTY, value=0.1) for _ in range(size)] for _ in range(size)]
        self.rules = CityRule.possible_values()

        if config == 0:
            self.population_growth_rate = self.rules[0].population_growth_rate
            self.infrastructure_development_rate = self.rules[0].infrastructure_development_rate
            self.environmental_impact_rate = self.rules[0].environmental_impact_rate
            self.environmental_conservation_rate = self.rules[0].environmental_conservation_rate
        elif config == 1:
            self.population_growth_rate = self.rules[1].population_growth_rate
            self.infrastructure_development_rate = self.rules[1].infrastructure_development_rate
            self.environmental_impact_rate = self.rules[1].environmental_impact_rate
            self.environmental_conservation_rate = self.rules[1].environmental_conservation_rate
        elif config == 2:
            self.population_growth_rate = self.rules[2].population_growth_rate
            self.infrastructure_development_rate = self.rules[2].infrastructure_development_rate
            self.environmental_impact_rate = self.rules[2].environmental_impact_rate
            self.environmental_conservation_rate = self.rules[2].environmental_conservation_rate
        elif config == 3:
            self.population_growth_rate = self.rules[3].population_growth_rate
            self.infrastructure_development_rate = self.rules[3].infrastructure_development_rate
            self.environmental_impact_rate = self.rules[3].environmental_impact_rate
            self.environmental_conservation_rate = self.rules[3].environmental_conservation_rate
        else:
            self.population_growth_rate = self.rules[4].population_growth_rate
            self.infrastructure_development_rate = self.rules[4].infrastructure_development_rate
            self.environmental_impact_rate = self.rules[4].environmental_impact_rate
            self.environmental_conservation_rate = self.rules[4].environmental_conservation_rate
        self.randomize_city()
    def randomize_city(self):
        for i in range(self.size):
            for j in range(self.size):
                self.grid[i][j] = CellState.random(self.config)

    def find_max_cell_type(self, residential_count, commercial_count, industrial_count, green_space_count):
        max_count = 0
        max_cell_type = ""

        if residential_count > max_count:
            max_count = residential_count
            max_cell_type = "R"
        if commercial_count > max_count:
            max_count = commercial_count
            max_cell_type = "C"
        if industrial_count > max_count:
            max_count = industrial_count
            max_cell_type = "I"
        if green_space_count > max_count:
            max_count = green_space_count
            max_cell_type = "G"

        return max_cell_type

    def find_min_cell_type(self, residential_count, commercial_count, industrial_count, green_space_count):
        min_count = float('inf')
        min_cell_type = ""

        if residential_count < min_count:
            min_count = residential_count
            min_cell_type = "R"
        if commercial_count < min_count:
            min_count = commercial_count
            min_cell_type = "C"
        if industrial_count < min_count:
            min_count = industrial_count
            min_cell_type = "I"
        if green_space_count < min_count:
            min_count = green_space_count
            min_cell_type = "G"

        return min_cell_type

    def select_parents(self):
        parent1_index = random.randint(0, len(self.rules) - 1)
        parent2_index = random.randint(0, len(self.rules) - 1)
        return self.rules[parent1_index], self.rules[parent2_index]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(CityRule.possible_values()) - 2)
        child = CityRule(0, 0, 0, 0)
        for i in range(crossover_point):
            child.population_growth_rate = parent1.population_growth_rate if random.random() < 0.5 else parent2.population_growth_rate
            child.environmental_conservation_rate = parent1.environmental_conservation_rate if random.random() < 0.5 else parent2.environmental_conservation_rate
            child.environmental_impact_rate = parent1.environmental_impact_rate if random.random() < 0.5 else parent2.environmental_impact_rate
            child.infrastructure_development_rate = parent1.infrastructure_development_rate if random.random() < 0.5 else parent2.infrastructure_development_rate
        self.population_growth_rate = child.population_growth_rate
        self.environmental_conservation_rate = child.environmental_conservation_rate
        self.environmental_impact_rate = child.environmental_impact_rate
        self.infrastructure_development_rate = child.infrastructure_development_rate

    def evolve(self):
        new_grid = [row[:] for row in self.grid]
        residential_count = self.count_residential_cells_state()
        commercial_count = self.count_commercial_cells_state()
        industrial_count = self.count_industrial_cells_state()
        green_space_count = self.count_green_cells_state()

        max_cell = self.find_max_cell_type(residential_count, commercial_count, industrial_count, green_space_count)
        min_cell = self.find_min_cell_type(residential_count, commercial_count, industrial_count, green_space_count)

        self.city_state = self.sum_city_state_values()

        if 0 < self.city_state < 70:
            for i in range(self.size):
                for j in range(self.size):
                    industrial_neighbors = self.count_industrial_neighbors(i, j)
                    commercial_neighbors = self.count_commercial_neighbors(i, j)
                    residential_neighbors = self.count_residential_neighbors(i, j)
                    green_neighbors = self.count_green_neighbors(i, j)
                    current_state = self.grid[i][j]
                    next_state = current_state

                    if current_state.state == CellState.State.EMPTY:
                        if residential_neighbors >= 3 and self.random_bool(self.population_growth_rate):
                            next_state = CellState(CellState.State.RESIDENTIAL, self.cell_state_value(CellState.State.RESIDENTIAL))
                        elif green_neighbors == 2 and self.random_bool(0.1):
                            next_state = CellState(CellState.State.GREEN_SPACE, self.cell_state_value(CellState.State.GREEN_SPACE))
                        elif self.random_bool(self.infrastructure_development_rate):
                            next_state = CellState(CellState.State.INDUSTRIAL, self.cell_state_value(CellState.State.INDUSTRIAL))
                        elif self.random_bool(self.environmental_conservation_rate):
                            next_state = CellState(CellState.State.COMMERCIAL, self.cell_state_value(CellState.State.COMMERCIAL))
                    elif current_state.state == CellState.State.RESIDENTIAL:
                        if residential_neighbors < 2 and self.random_bool(0.1) or residential_neighbors > 10:
                            next_state = CellState(CellState.State.EMPTY, self.cell_state_value(CellState.State.EMPTY))
                        elif commercial_neighbors > 0 and self.random_bool(0.1):
                            next_state = CellState(CellState.State.COMMERCIAL, self.cell_state_value(CellState.State.COMMERCIAL))
                        elif self.random_bool(self.population_growth_rate):
                            next_state = CellState(CellState.State.RESIDENTIAL, self.cell_state_value(CellState.State.RESIDENTIAL))
                    elif current_state.state == CellState.State.COMMERCIAL:
                        if commercial_neighbors < 2 or commercial_neighbors > 4:
                            next_state = CellState(CellState.State.EMPTY, self.cell_state_value(CellState.State.EMPTY))
                        elif residential_neighbors > 0 and self.random_bool(self.population_growth_rate):
                            next_state = CellState(CellState.State.RESIDENTIAL, self.cell_state_value(CellState.State.RESIDENTIAL))
                        elif self.random_bool(self.infrastructure_development_rate):
                            next_state = CellState(CellState.State.COMMERCIAL, self.cell_state_value(CellState.State.COMMERCIAL))
                    elif current_state.state == CellState.State.INDUSTRIAL:
                        if industrial_neighbors < 2 or industrial_neighbors > 4:
                            next_state = CellState(CellState.State.EMPTY, self.cell_state_value(CellState.State.EMPTY))
                        elif residential_neighbors > 0:
                            next_state = CellState(CellState.State.RESIDENTIAL, self.cell_state_value(CellState.State.RESIDENTIAL))
                        elif self.random_bool(self.environmental_impact_rate):
                            next_state = CellState(CellState.State.INDUSTRIAL, self.cell_state_value(CellState.State.INDUSTRIAL))
                    elif current_state.state == CellState.State.GREEN_SPACE:
                        if green_neighbors >= 5 and self.random_bool(0.1):
                            next_state = CellState(CellState.State.RESIDENTIAL, self.cell_state_value(CellState.State.RESIDENTIAL))
                        elif green_neighbors <= 1 and self.random_bool(0.1):
                            next_state = CellState(CellState.State.EMPTY, self.cell_state_value(CellState.State.EMPTY))
                        elif self.random_bool(self.environmental_conservation_rate):
                            next_state = CellState(CellState.State.GREEN_SPACE, self.cell_state_value(CellState.State.GREEN_SPACE))

                    new_grid[i][j] = next_state

        self.grid = new_grid

        if self.city_state < 0:
            for i in range(self.size):
                for j in range(self.size):
                    current_state = self.grid[i][j]
                    next_state = current_state
                    if current_state.symbol == min_cell and self.random_bool(0.7):
                        next_state = CellState(self.cell_type_val(max_cell), self.cell_state_value(self.cell_type_val(max_cell)))
                    new_grid[i][j] = next_state

            self.grid = new_grid
            self.reset_counter += 1

            if self.reset_counter >= 20:
                parent1, parent2 = self.select_parents()
                self.crossover(parent1, parent2)
                self.reset_counter = 0
        else:
            for i in range(self.size):
                for j in range(self.size):
                    current_state = self.grid[i][j]
                    next_state = current_state
                    if current_state.symbol == max_cell and self.random_bool(0.4):
                        next_state = CellState(CellState.State.EMPTY, self.cell_state_value(CellState.State.EMPTY))
                    new_grid[i][j] = next_state

            self.grid = new_grid
            self.reset_counter += 1

            if self.reset_counter >= 20:
                parent1, parent2 = self.select_parents()
                self.crossover(parent1, parent2)
                self.reset_counter = 0

    def cell_type_val(self, symbol):
        if symbol == "R":
            return CellState.State.RESIDENTIAL
        elif symbol == "C️":
            return CellState.State.COMMERCIAL
        elif symbol == "I":
            return CellState.State.INDUSTRIAL
        elif symbol == "G":
            return CellState.State.GREEN_SPACE
        else:
            return CellState.State.EMPTY

    def cell_state_value(self, state):
        if self.config == 0:
            if state == CellState.State.EMPTY:
                return 0.1
            elif state == CellState.State.RESIDENTIAL:
                return 0.1
            elif state == CellState.State.COMMERCIAL:
                return 0.1
            elif state == CellState.State.INDUSTRIAL:
                return 0.1
            elif state == CellState.State.GREEN_SPACE:
                return 0.1
        elif self.config == 1:
            if state == CellState.State.EMPTY:
                return 0.1
            elif state == CellState.State.RESIDENTIAL:
                return 0.5
            elif state == CellState.State.COMMERCIAL:
                return 0.2
            elif state == CellState.State.INDUSTRIAL:
                return -0.2
            elif state == CellState.State.GREEN_SPACE:
                return 0.3
        elif self.config == 2:
            if state == CellState.State.EMPTY:
                return 0.1
            elif state == CellState.State.RESIDENTIAL:
                return -0.2
            elif state == CellState.State.COMMERCIAL:
                return 0.5
            elif state == CellState.State.INDUSTRIAL:
                return 0.1
            elif state == CellState.State.GREEN_SPACE:
                return -0.2
        elif self.config == 3:
            if state == CellState.State.EMPTY:
                return 0.1
            elif state == CellState.State.RESIDENTIAL:
                return -0.1
            elif state == CellState.State.COMMERCIAL:
                return 0.1
            elif state == CellState.State.INDUSTRIAL:
                return 0.5
            elif state == CellState.State.GREEN_SPACE:
                return -0.2
        else:
            if state == CellState.State.EMPTY:
                return 0.1
            elif state == CellState.State.RESIDENTIAL:
                return 0.1
            elif state == CellState.State.COMMERCIAL:
                return 0.1
            elif state == CellState.State.INDUSTRIAL:
                return 0.1
            elif state == CellState.State.GREEN_SPACE:
                return 0.1

    def random_bool(self, probability):
        return random.random() < probability

    def count_living_neighbors(self, x, y):
        count = 0
        for i in range(-1,2):
            for j in range(-1,2):
                row = (x + i + self.size) % self.size
                col = (y +j + self.size) % self.size
                if not (i == 0 and j ==0) and self.grid[row][col].is_living():
                    count += 1
        return count

    def count_industrial_cells(self):
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j].state == CellState.State.INDUSTRIAL:
                    count += 1
        return count

    def count_empty_cells(self):
        count =0
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j].state == CellState.State.EMPTY:
                    count+=1
        return count

    def count_industrial_cells_state(self):
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j].state == CellState.State.INDUSTRIAL:
                    count += self.grid[i][j].value
        return count

    def count_industrial_neighbors(self, x, y):
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                row = (x + i + self.size) % self.size
                col = (y + j + self.size) % self.size
                if not (i == 0 and j == 0) and self.grid[row][col].state == CellState.State.INDUSTRIAL:
                    count +=1
        return count


    def count_residential_cells(self):
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j].state == CellState.State.RESIDENTIAL:
                    count += 1
        return count

    def count_residential_cells_state(self):
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j].state == CellState.State.RESIDENTIAL:
                    count += self.grid[i][j].value
        return count

    def count_residential_neighbors(self, x, y):
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                row = (x + i + self.size) % self.size
                col = (y + j + self.size) % self.size
                if (i, j) != (0, 0) and self.grid[row][col].state == CellState.State.RESIDENTIAL:
                    count += 1
        return count

    def count_commercial_cells_state(self):
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j].state == CellState.State.COMMERCIAL:
                    count += self.grid[i][j].value
        return count


    def count_commercial_cells(self):
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j].state == CellState.State.COMMERCIAL:
                    count += 1
        return count

    def count_commercial_neighbors(self, x, y):
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                row = (x + i + self.size) % self.size
                col = (y + j + self.size) % self.size
                if (i, j) != (0, 0) and self.grid[row][col].state == CellState.State.COMMERCIAL:
                    count += 1
        return count

    def count_green_cells_state(self):
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j].state == CellState.State.GREEN_SPACE:
                    count += self.grid[i][j].value
        return count

    def count_green_cells(self):
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j].state == CellState.State.GREEN_SPACE:
                    count += 1
        return count

    def count_green_neighbors(self, x, y):
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                row = (x + i + self.size) % self.size
                col = (y + j + self.size) % self.size
                if (i, j) != (0, 0) and self.grid[row][col].state == CellState.State.GREEN_SPACE:
                    count += 1
        return count

    def sum_city_state_values(self):
        total = 0
        for i in range(self.size):
            for j in range(self.size):
                total += self.grid[i][j].value
        return total

    def print_city(self, file_path, generation):
        residential_count = self.count_residential_cells()
        commercial_count = self.count_commercial_cells()
        industrial_count = self.count_industrial_cells()
        green_space_count = self.count_green_cells()
        city_state = self.sum_city_state_values()

        with open(file_path, "a") as file:
            file.write(f"\nStatistics:\n")
            file.write(f"Generation: {generation}\n")
            file.write(f"City State Value: {city_state}\n")
            file.write(f"Residential: {residential_count}\n")
            file.write(f"Commercial: {commercial_count}\n")
            file.write(f"Industrial: {industrial_count}\n")
            file.write(f"Green Space: {green_space_count}\n")
            file.write(f"POPULATION GROWTH RATE: {self.population_growth_rate}\n")
            file.write(f"INFRASTRUCTURE DEVELOPMENT RATE: {self.infrastructure_development_rate}\n")
            file.write(f"ENVIRONMENTAL CONSERVATION RATE: {self.environmental_conservation_rate}\n")
            file.write(f"ENVIRONMENTAL IMPACT RATE: {self.environmental_impact_rate}\n")
            file.write(f"DEVELOPMENT RATE MUTATION: {self.reset_counter}\n\n")

        print(f"\nStatistics:")
        print(f"City State Value: {city_state}")
        print(f"Residential: {residential_count}")
        print(f"Commercial: {commercial_count}")
        print(f"Industrial: {industrial_count}")
        print(f"Green Space: {green_space_count}")
        print(f"POPULATION GROWTH RATE: {self.population_growth_rate}")
        print(f"INFRASTRUCTURE DEVELOPMENT RATE: {self.infrastructure_development_rate}")
        print(f"ENVIRONMENTAL CONSERVATION RATE: {self.environmental_conservation_rate}")
        print(f"ENVIRONMENTAL IMPACT RATE: {self.environmental_impact_rate}")
        print(f"DEVELOPMENT RATE MUTATION: {self.reset_counter}")

def main():
    size = 20
    welcome_message = """
    ***********************************
    *           CitySim              *
    ***********************************
    Input configuration:
    0 - Default Configuration
    1 - Residential Configuration
    2 - Commercial Zone Configuration
    3 - Industrial Zone Configuration
    """
    choose_generation_message = """
    ***********************************
    *           CitySim              *
    ***********************************
    Choose the amount of generations you would like to simulate (recommended = 10000):
    """
    choose_time_message = """
    ***********************************
    *           CitySim              *
    ***********************************
    Choose the amount of time between each generation display (in milliseconds):
    """
    end_of_simulation_message = """
    ***********************************
    *           CitySim              *
    ***********************************
    The Simulation is done! The Generation by Generation progression of the city growth can be found in the generated results.txt file
    These are the final stable growth values:
    """

    print(welcome_message)
    config = int(input("Enter configuration (0-3): "))

    print(choose_generation_message)
    generations = int(input("Enter the number of generations: "))

    print(choose_time_message)
    time_ms = int(input("Enter the time between generations (in milliseconds): "))

    city = CitySimulation(size, config, True)
    for generation in range(generations):
        print(f"Config {config}:")
        print(f"Generation {generation + 1}:")
        city.print_city(f"results_{config}.txt", generation)
        print()
        city.evolve()
        import time
        time.sleep(time_ms / 1000)

    print(end_of_simulation_message)
    print(f"Config {config}")
    print(f"POPULATION GROWTH RATE: {city.population_growth_rate}")
    print(f"INFRASTRUCTURE DEVELOPMENT RATE: {city.infrastructure_development_rate}")
    print(f"ENVIRONMENTAL CONSERVATION RATE: {city.environmental_conservation_rate}")
    print(f"ENVIRONMENTAL IMPACT RATE: {city.environmental_impact_rate}")
    print(f"DEVELOPMENT RATE MUTATION: {city.reset_counter}")

if __name__ == "__main__":
    main()