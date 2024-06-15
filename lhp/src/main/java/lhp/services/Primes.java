package lhp.services;

import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.Application;
import jakarta.ws.rs.core.MediaType;

import java.util.ArrayList;
import java.util.List;

@Path("/primes")
public class Primes extends Application {
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public List<Integer> primes() {
        List<Integer> result = new ArrayList<>();
        for (int n = 2; n <= 100; n++)
            if (isPrime(n)) result.add(n);
        return result;
    }

    private boolean isPrime(int n) {
        if (n < 2) return false;
        if (n == 2) return true;
        for (int j = 2; j < Math.sqrt(n) + 1; j++)
            if (n % j == 0) return false;
        return true;
    }
}
