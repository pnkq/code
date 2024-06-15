package lhp.services;

import jakarta.ws.rs.*;
import jakarta.ws.rs.core.Application;
import jakarta.ws.rs.core.MediaType;

import java.util.ArrayList;
import java.util.List;

@Path("/primes")
public class Primes extends Application {
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public List<Integer> primes(@QueryParam("n") String max) {
        int bound = 100;
        if (max != null)
            bound = Integer.parseInt(max);
        List<Integer> result = new ArrayList<>();
        for (int n = 2; n <= bound; n++)
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
